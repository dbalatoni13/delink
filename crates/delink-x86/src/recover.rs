//! x86 (32-bit) relocation recovery from linked PE code.
//!
//! Handles:
//!   * `E8 rel32`    (call rel32)   → IMAGE_REL_I386_REL32 at offset 1
//!   * `E9 rel32`    (jmp rel32)    → IMAGE_REL_I386_REL32 at offset 1
//!   * `0F 8x rel32` (jcc rel32)   → IMAGE_REL_I386_REL32 at offset 2
//!
//! Most 32-bit absolute pointer fixups come from the PE base-relocation table.
//! The MSVC SEH `fs:[0]` access is an exception: its encoded displacement is
//! zero in the linked image, but the source object relocates it to
//! `__except_list`.
//!
//! Intra-function branches are skipped. Unresolved targets are counted.

use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, Mnemonic, Register};
use tracing::trace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocKind {
    /// IMAGE_REL_I386_REL32 — 32-bit PC-relative (calls, jumps).
    Rel32,
    /// IMAGE_REL_I386_DIR32 — MSVC's `__except_list` at `fs:[0]`.
    Dir32,
}

#[derive(Debug, Clone)]
pub struct RecoveredReloc {
    /// Byte offset within the function bytes where the fixup field lives.
    pub offset: u64,
    /// Instruction address (fn_va + offset of instruction start).
    pub pc: u64,
    pub kind: RelocKind,
    /// Symbol name the reloc targets.
    pub target: String,
    /// Addend relative to the symbol (0 = target is exactly the symbol start).
    pub addend: i64,
}

#[derive(Debug, Default)]
pub struct RecoveryDiagnostics {
    pub instructions: usize,
    pub decode_failures: usize,
    pub calls_resolved: usize,
    pub calls_unresolved: usize,
    /// Always 0 for x86 (no RIP-relative addressing); kept for API symmetry.
    pub rip_refs_unresolved: usize,
}

pub struct RecoveryOutput {
    pub relocs: Vec<RecoveredReloc>,
    pub diag: RecoveryDiagnostics,
    /// Byte offsets (within the function) of the `F3` prefix of each `rep ret`
    /// (`F3 C3`) instruction. The emitter can strip these to plain `ret`.
    pub rep_ret_offsets: Vec<u64>,
}

/// Trait that callers implement to resolve VAs to symbol names.
pub trait SymbolResolver {
    /// Resolve a code target (call/jmp destination) → (symbol_name, addend).
    fn resolve_code(&self, va: u64) -> Option<(String, i64)>;
    /// Resolve a data reference → (symbol_name, addend).
    fn resolve_data(&self, va: u64) -> Option<(String, i64)>;
    /// Resolve a zero displacement through FS, when the target ABI gives it a
    /// named COFF symbol (MSVC x86 uses `__except_list`).
    fn resolve_fs_zero(&self) -> Option<String> {
        None
    }
    /// Returns true if `target_va` is inside the current function body.
    fn is_intra_function(&self, fn_va: u64, fn_size: u64, target_va: u64) -> bool {
        target_va >= fn_va && target_va < fn_va + fn_size
    }
}

/// Walk `fn_bytes` starting at `fn_va`, synthesise COFF relocations for
/// all direct calls and jumps that leave the function.
pub fn recover<R: SymbolResolver>(
    fn_bytes: &[u8],
    fn_va: u64,
    fn_size: u64,
    resolver: &R,
) -> Result<RecoveryOutput> {
    // For 32-bit PE, fn_va fits in 32 bits; pass the low 32 bits as IP.
    let mut decoder = Decoder::with_ip(32, fn_bytes, fn_va & 0xFFFF_FFFF, DecoderOptions::NONE);
    let mut insn = Instruction::default();

    let mut out = RecoveryOutput {
        relocs: Vec::new(),
        diag: RecoveryDiagnostics::default(),
        rep_ret_offsets: Vec::new(),
    };

    while decoder.can_decode() {
        decoder.decode_out(&mut insn);
        out.diag.instructions += 1;

        if insn.is_invalid() {
            out.diag.decode_failures += 1;
            continue;
        }

        let pc = insn.ip();
        let insn_offset = pc - (fn_va & 0xFFFF_FFFF);
        let insn_len = insn.len() as u64;

        // `rep ret` (F3 C3): a 2-byte near return carrying a redundant REP
        // prefix. Record the `F3` byte so the emitter can drop it to plain `ret`.
        if insn.mnemonic() == Mnemonic::Ret
            && insn_len == 2
            && fn_bytes.get(insn_offset as usize) == Some(&0xF3)
        {
            out.rep_ret_offsets.push(insn_offset);
        }

        // A linked PE retains the zero displacement in `fs:[0]`, so neither
        // IDA nor the PE base-relocation table can recover this COFF symbol.
        // Require an explicit 32-bit displacement and no base/index register:
        // `fs:[eax]` and `fs:[4]` must not be treated as `__except_list`.
        if insn.memory_segment() == Register::FS
            && insn.memory_base() == Register::None
            && insn.memory_index() == Register::None
            && insn.memory_displacement64() == 0
        {
            let offsets = decoder.get_constant_offsets(&insn);
            if offsets.displacement_size() == 4 {
                if let Some(target) = resolver.resolve_fs_zero() {
                    out.relocs.push(RecoveredReloc {
                        offset: insn_offset + offsets.displacement_offset() as u64,
                        pc,
                        kind: RelocKind::Dir32,
                        target,
                        addend: 0,
                    });
                }
            }
        }

        // Only direct near branches (call rel32 / jmp rel32 / jcc rel32).
        // rel8 branches are 2 bytes; rel32 are 5 (E8/E9) or 6 (0F 8x) bytes.
        match insn.flow_control() {
            FlowControl::Call
            | FlowControl::UnconditionalBranch
            | FlowControl::ConditionalBranch
                if insn_len >= 5 =>
            {
                // near_branch32() gives the absolute 32-bit target VA.
                let target_va = insn.near_branch32() as u64;

                if !resolver.is_intra_function(fn_va, fn_size, target_va) {
                    // The rel32 field is always the last 4 bytes of these instructions.
                    let rel32_off = insn_len - 4;

                    match resolver.resolve_code(target_va) {
                        Some((sym, addend)) => {
                            out.relocs.push(RecoveredReloc {
                                offset: insn_offset + rel32_off,
                                pc,
                                kind: RelocKind::Rel32,
                                target: sym,
                                addend,
                            });
                            out.diag.calls_resolved += 1;
                        }
                        None => {
                            trace!("{:#x}: unresolved call/jmp target {:#x}", pc, target_va);
                            out.diag.calls_unresolved += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoSymbols(bool);

    impl SymbolResolver for NoSymbols {
        fn resolve_code(&self, _: u64) -> Option<(String, i64)> {
            None
        }

        fn resolve_data(&self, _: u64) -> Option<(String, i64)> {
            None
        }

        fn resolve_fs_zero(&self) -> Option<String> {
            self.0.then(|| "__except_list".to_string())
        }
    }

    #[test]
    fn recovers_msvc_seh_exception_list_in_loads_and_stores() {
        // mov eax, fs:[0]; mov fs:[0], esp; mov eax, fs:[4]; mov eax, gs:[0]
        let bytes = [
            0x64, 0xa1, 0, 0, 0, 0, 0x64, 0x89, 0x25, 0, 0, 0, 0, 0x64, 0xa1, 4, 0, 0, 0, 0x65,
            0xa1, 0, 0, 0, 0,
        ];
        let result = recover(&bytes, 0x704230, bytes.len() as u64, &NoSymbols(true)).unwrap();
        assert_eq!(result.relocs.len(), 2);
        for (reloc, offset) in result.relocs.iter().zip([2, 9]) {
            assert_eq!(reloc.offset, offset);
            assert_eq!(reloc.kind, RelocKind::Dir32);
            assert_eq!(reloc.target, "__except_list");
            assert_eq!(reloc.addend, 0);
        }
        assert!(
            recover(&bytes, 0x704230, bytes.len() as u64, &NoSymbols(false))
                .unwrap()
                .relocs
                .is_empty()
        );
    }
}
