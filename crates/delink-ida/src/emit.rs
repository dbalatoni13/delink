//! Object emitter for the IDA importer.
//!
//! Produces one COFF `.obj` (or ELF `.o`) per group of functions plus a single
//! `__shared_data` object carrying the data/const/bss sections.  For x86 /
//! x86-64 it runs iced-x86 relocation recovery ([`delink_x86`] /
//! [`delink_x86_64`]) for rel32 calls/jumps and RIP-relative references, and
//! uses IDA's fixup table for absolute pointers.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use object::write::{Mangling, Object, Relocation, SectionId, Symbol, SymbolId, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags, SymbolKind,
    SymbolScope,
};
use rayon::prelude::*;

use crate::idapro_json::{IdaproJson, ObjectGroup};
use crate::resolver::{IdaSymbols, SYM_BSS_START, SYM_CONST_START, SYM_DATA_START};
use crate::{Function, IdaArch, IdaModel, PeImage, SegClass};

/// Read `len` bytes at `rva` from the original binary, zero-padding any tail not
/// backed by raw section data (e.g. virtual-size padding).
fn read_padded(pe: &PeImage, rva: u64, len: usize) -> Vec<u8> {
    if let Some(b) = pe.data_at_rva(rva, len) {
        return b.to_vec();
    }
    let mut out = vec![0u8; len];
    if let Some(sec) = pe.section_for_rva(rva) {
        let off = (rva - sec.rva) as usize;
        let n = sec.data.len().saturating_sub(off).min(len);
        if n > 0 {
            out[..n].copy_from_slice(&sec.data[off..off + n]);
        }
    }
    out
}

/// REL32 fields are next-instruction-relative; the object writer (and the ELF
/// S+A−P convention) reference the field start, so subtract the 4-byte field
/// width from the recovered addend (same adjustment as the PE/Mach-O emitters).
const REL32_FIELD_BYTES: i64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Coff,
    Elf,
}

impl OutputFormat {
    pub fn binary_format(self) -> BinaryFormat {
        match self {
            OutputFormat::Coff => BinaryFormat::Coff,
            OutputFormat::Elf => BinaryFormat::Elf,
        }
    }
    /// Default output object extension.
    pub fn ext(self) -> &'static str {
        match self {
            OutputFormat::Coff => "obj",
            OutputFormat::Elf => "o",
        }
    }
    /// Pick a sensible default from IDA's reported file type.
    pub fn default_for_filetype(filetype: &str) -> OutputFormat {
        match filetype {
            "ELF" | "MACHO" => OutputFormat::Elf,
            _ => OutputFormat::Coff, // PE / COFF / unknown
        }
    }
}

#[derive(Debug, Default)]
pub struct EmitStats {
    pub text_bytes: u64,
    pub data_bytes: u64,
    pub const_bytes: u64,
    pub bss_bytes: u64,
    pub instructions: usize,
    pub local_symbols: usize,
    pub undef_symbols: usize,
    pub relocations: usize,
    pub unresolved_calls: usize,
    pub unresolved_rip: usize,
}

#[derive(Debug, Default)]
pub struct SharedDataStats {
    pub data_bytes: u64,
    pub const_bytes: u64,
    pub bss_bytes: u64,
    pub relocations: usize,
}

#[derive(Debug)]
pub struct CuOutcome {
    pub cu_name: String,
    pub file: std::path::PathBuf,
    pub result: std::result::Result<EmitStats, String>,
}

#[derive(Debug)]
struct OwnedDataRange {
    range: Range<u64>,
    anchor: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Split driven by an `idapro.json` grouping. Function addresses sharing a key
/// are emitted into one object named exactly by that key; all function metadata
/// is resolved from `model`.
pub fn split_by_groups(
    model: &IdaModel,
    pe: &PeImage,
    symbols: &IdaSymbols,
    groups: &IdaproJson,
    out_dir: &Path,
    format: OutputFormat,
) -> Result<Vec<CuOutcome>> {
    if model.arch == IdaArch::Other {
        return Err(anyhow!(
            "relocation recovery is only implemented for x86 / x86-64 (procname {:?})",
            model.procname
        ));
    }
    std::fs::create_dir_all(out_dir).with_context(|| format!("create {}", out_dir.display()))?;

    // Build the model lookup once. The default grouping has one object per
    // function, so rebuilding this map inside every object emission would be
    // quadratic in the number of functions.
    let functions_by_start: HashMap<u64, &Function> =
        model.functions.iter().map(|f| (f.start, f)).collect();

    let groups = expand_function_ranges(model, groups)?;
    validate_data_ranges(model, &groups)?;
    let owned_ranges = owned_data_ranges(&groups);

    let group_vec: Vec<(&String, &ObjectGroup)> = groups.iter().collect();
    let outcomes = group_vec
        .par_iter()
        .map(|(file_name, group)| {
            let file = out_dir.join(file_name.as_str());
            let result = emit_object(
                model,
                pe,
                symbols,
                &functions_by_start,
                group,
                &owned_ranges,
                format,
                &file,
            )
            .map_err(|e| format!("{e:#}"));
            CuOutcome {
                cu_name: (*file_name).clone(),
                file,
                result,
            }
        })
        .collect();
    Ok(outcomes)
}

/// Emit the `__shared_data` object carrying data / const / bss sections.
pub fn emit_shared(
    model: &IdaModel,
    pe: &PeImage,
    symbols: &IdaSymbols,
    out_path: &Path,
    format: OutputFormat,
) -> Result<SharedDataStats> {
    emit_shared_excluding(model, pe, symbols, &[], &[], out_path, format)
}

/// Emit data not assigned to an object in `idapro.json` into `__shared_data`.
pub fn emit_shared_for_groups(
    model: &IdaModel,
    pe: &PeImage,
    symbols: &IdaSymbols,
    groups: &IdaproJson,
    out_path: &Path,
    format: OutputFormat,
) -> Result<SharedDataStats> {
    validate_data_ranges(model, groups)?;
    let owned_ranges = owned_data_ranges(groups);
    let excluded: Vec<Range<u64>> = owned_ranges
        .iter()
        .map(|owned| owned.range.clone())
        .collect();
    emit_shared_excluding(
        model,
        pe,
        symbols,
        &excluded,
        &owned_ranges,
        out_path,
        format,
    )
}

fn emit_shared_excluding(
    model: &IdaModel,
    pe: &PeImage,
    symbols: &IdaSymbols,
    excluded: &[Range<u64>],
    owned_ranges: &[OwnedDataRange],
    out_path: &Path,
    format: OutputFormat,
) -> Result<SharedDataStats> {
    let (arch, endian) = obj_arch(model);
    let mut obj = Object::new(format.binary_format(), arch, endian);
    if format == OutputFormat::Coff && model.arch == IdaArch::X86 {
        obj.set_mangling(Mangling::Coff);
    }
    let mut defined: HashMap<String, SymbolId> = HashMap::new();
    let mut undef: HashMap<String, SymbolId> = HashMap::new();
    let mut stats = SharedDataStats::default();

    // Data relocations are staged and emitted in a second pass, after every
    // section's start symbol and named variables exist, so a pointer into a
    // later section (or a not-yet-defined variable) still binds locally.
    struct PendingData {
        sid: SectionId,
        offset: u64,
        name: String,
        addend: i64,
        size: u32,
    }
    let mut pending: Vec<PendingData> = Vec::new();

    for sec in &model.sections {
        let (kind, name, start_sym, bytes_field): (SectionKind, &str, &str, &mut u64) =
            match sec.class {
                SegClass::Data => (
                    SectionKind::Data,
                    data_name(format),
                    SYM_DATA_START,
                    &mut stats.data_bytes,
                ),
                SegClass::Const => (
                    SectionKind::ReadOnlyData,
                    const_name(format),
                    SYM_CONST_START,
                    &mut stats.const_bytes,
                ),
                SegClass::Bss => (
                    SectionKind::UninitializedData,
                    bss_name(format),
                    SYM_BSS_START,
                    &mut stats.bss_bytes,
                ),
                _ => continue,
            };

        // All configured ranges are removed from the shared object. This is
        // important for logical `bss` ranges that originate in an IDA DATA
        // segment: their bytes must not remain in shared `.data`.
        let fragments = subtract_ranges(sec.start..sec.end, excluded);
        for fragment in fragments {
            let sid = obj.add_section(Vec::new(), name.as_bytes().to_vec(), kind);
            let size = fragment.end - fragment.start;
            if kind == SectionKind::UninitializedData {
                obj.section_mut(sid).append_bss(size, 16);
            } else {
                let rva = fragment.start.wrapping_sub(model.image_base);
                let mut bytes = read_padded(pe, rva, size as usize);
                for r in symbols.relocs_in(fragment.clone()) {
                    if abs_flags(format, model.arch, r.size).is_none() {
                        continue;
                    }
                    let off = (r.addr - fragment.start) as usize;
                    let w = r.size as usize;
                    if off + w > bytes.len() {
                        return Err(anyhow!(
                            "data split boundary intersects the relocation at {:#x}",
                            r.addr
                        ));
                    }
                    if let Some((name, addend)) =
                        resolve_split_data(symbols, owned_ranges, r.target)
                    {
                        bytes[off..off + w].fill(0);
                        pending.push(PendingData {
                            sid,
                            offset: off as u64,
                            name,
                            addend,
                            size: r.size,
                        });
                    }
                }
                obj.append_section_data(sid, &bytes, 16);
            }

            if is_first_class_address(model, sec.class, fragment.start) {
                let id = obj.add_symbol(Symbol {
                    name: start_sym.as_bytes().to_vec(),
                    value: 0,
                    size: 0,
                    kind: SymbolKind::Data,
                    scope: SymbolScope::Dynamic,
                    weak: false,
                    section: SymbolSection::Section(sid),
                    flags: SymbolFlags::None,
                });
                defined.insert(start_sym.to_string(), id);
            }

            for (va, var) in symbols.variables.range(fragment.clone()) {
                let scope = if var.public {
                    SymbolScope::Dynamic
                } else {
                    SymbolScope::Compilation
                };
                let id = obj.add_symbol(Symbol {
                    name: sanitize_symbol_name(&var.name),
                    value: va - fragment.start,
                    size: 0,
                    kind: SymbolKind::Data,
                    scope,
                    weak: false,
                    section: SymbolSection::Section(sid),
                    flags: SymbolFlags::None,
                });
                defined.entry(var.name.clone()).or_insert(id);
            }
            *bytes_field += size;
        }
    }

    for p in pending {
        let Some(flags) = abs_flags(format, model.arch, p.size) else {
            continue;
        };
        let sym = match defined.get(&p.name) {
            Some(id) => *id,
            None => resolve_or_add_undef(&mut obj, &mut undef, &p.name, format),
        };
        obj.add_relocation(
            p.sid,
            Relocation {
                offset: p.offset,
                symbol: sym,
                addend: p.addend,
                flags,
            },
        )
        .with_context(|| format!("add data reloc at {:#x}", p.offset))?;
        stats.relocations += 1;
    }

    let bytes = obj.write().context("serialize shared object")?;
    write_file(out_path, &bytes)?;
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Per-object emit
// ---------------------------------------------------------------------------

fn emit_object(
    model: &IdaModel,
    pe: &PeImage,
    symbols: &IdaSymbols,
    functions_by_start: &HashMap<u64, &Function>,
    group: &ObjectGroup,
    owned_ranges: &[OwnedDataRange],
    format: OutputFormat,
    out_path: &Path,
) -> Result<EmitStats> {
    // `idapro.json` contains grouping only. Resolve every configured address
    // through the authoritative delink model for name, bounds, and visibility.
    let mut seen = HashSet::new();
    let mut funcs = Vec::with_capacity(group.functions.len());
    for &address in &group.functions {
        if !seen.insert(address) {
            return Err(anyhow!("duplicate function address {address:#x} in group"));
        }
        let f = functions_by_start.get(&address).ok_or_else(|| {
            anyhow!("idapro address {address:#x} is not a function start in delink.json")
        })?;
        if f.size() == 0 {
            return Err(anyhow!(
                "function '{}' at {address:#x} has zero size in delink.json",
                f.name
            ));
        }
        funcs.push(*f);
    }
    funcs.sort_by_key(|f| f.start);
    if funcs.is_empty() && group.rdata.is_empty() && group.data.is_empty() && group.bss.is_empty() {
        return Err(anyhow!("group has no functions or data ranges"));
    }

    let (arch, endian) = obj_arch(model);
    let mut obj = Object::new(format.binary_format(), arch, endian);
    if format == OutputFormat::Coff && model.arch == IdaArch::X86 {
        obj.set_mangling(Mangling::Coff);
    }
    let text_name = text_name(format);
    let sid = obj.add_section(Vec::new(), text_name.as_bytes().to_vec(), SectionKind::Text);

    let mut local: HashMap<String, SymbolId> = HashMap::new();
    let mut undef: HashMap<String, SymbolId> = HashMap::new();
    let mut stats = EmitStats::default();

    // Relocations are staged (by target *name*) and emitted in a second pass,
    // after every function symbol in the group exists, so an intra-object
    // forward call binds to the local definition rather than a stray undef.
    struct Pending {
        sid: SectionId,
        offset: u64,
        sym: String,
        addend: i64,
        flags: RelocationFlags,
    }
    let mut pending: Vec<Pending> = Vec::new();
    let rel32 = rel32_flags(format, model.arch);

    for f in &funcs {
        let name = &f.name;
        let start = f.start;
        let end = f.end;
        let size = f.size();
        let rva = start.wrapping_sub(model.image_base);
        let Some(orig) = pe.data_at_rva(rva, size as usize) else {
            tracing::warn!(
                "function '{name}' at {start:#x} (rva {rva:#x}) not backed by the input binary; skipping"
            );
            continue;
        };
        let mut bytes = orig.to_vec();
        stats.text_bytes += size;

        // 1) iced-x86 recovery → rel32 relocations.
        let recovered = recover(model, &bytes, start, size, symbols, owned_ranges)?;
        stats.instructions += recovered.instructions;
        stats.unresolved_calls += recovered.unresolved_calls;
        stats.unresolved_rip += recovered.unresolved_rip;
        for r in &recovered.relocs {
            let off = r.offset as usize;
            if off + 4 <= bytes.len() {
                bytes[off..off + 4].fill(0);
            }
        }

        // 2) IDA fixup table → absolute relocations within this function.
        let mut abs: Vec<(u64, String, i64, RelocationFlags)> = Vec::new();
        for r in symbols.relocs_in(start..end) {
            let Some(flags) = abs_flags(format, model.arch, r.size) else {
                continue;
            };
            let off = (r.addr - start) as usize;
            let w = r.size as usize;
            if off + w > bytes.len() {
                continue;
            }
            if let Some((tname, addend)) = resolve_split_data(symbols, owned_ranges, r.target) {
                bytes[off..off + w].fill(0);
                abs.push((off as u64, tname, addend, flags));
            }
        }

        let fn_off = obj.append_section_data(sid, &bytes, 1);

        let scope = if f.public {
            SymbolScope::Dynamic
        } else {
            SymbolScope::Compilation
        };
        let sym_id = obj.add_symbol(Symbol {
            name: sanitize_symbol_name(name),
            value: fn_off,
            size,
            kind: SymbolKind::Text,
            scope,
            weak: false,
            section: SymbolSection::Section(sid),
            flags: SymbolFlags::None,
        });
        local.insert(name.clone(), sym_id);

        for (off, tname, addend, flags) in abs {
            pending.push(Pending {
                sid,
                offset: fn_off + off,
                sym: tname,
                addend,
                flags,
            });
        }
        for r in &recovered.relocs {
            pending.push(Pending {
                sid,
                offset: fn_off + r.offset,
                sym: r.target.clone(),
                addend: r.addend - REL32_FIELD_BYTES,
                flags: rel32,
            });
        }
    }

    // Emit the configured initialized-data contributions into this object.
    for (class, configured) in [
        (SegClass::Const, group.rdata.as_slice()),
        (SegClass::Data, group.data.as_slice()),
        (SegClass::Bss, group.bss.as_slice()),
    ] {
        for configured_range in configured {
            let range = configured_range.range();
            let _section = model
                .sections
                .iter()
                .find(|section| {
                    (if class == SegClass::Bss {
                        matches!(section.class, SegClass::Bss | SegClass::Data)
                    } else {
                        section.class == class
                    }) && section.start <= range.start
                        && range.end <= section.end
                })
                .expect("data ranges were validated before emission");
            let (kind, section_name) = match class {
                SegClass::Const => (SectionKind::ReadOnlyData, const_name(format)),
                SegClass::Data => (SectionKind::Data, data_name(format)),
                SegClass::Bss => (SectionKind::UninitializedData, bss_name(format)),
                _ => unreachable!("only data classes are emitted here"),
            };
            let data_sid = obj.add_section(Vec::new(), section_name.as_bytes().to_vec(), kind);
            let size = range.end - range.start;
            if class == SegClass::Bss {
                // A logical BSS range is always zero-filled, even when IDA
                // presents its virtual address range as part of `.data`.
                obj.section_mut(data_sid).append_bss(size, 16);
            } else {
                let rva = range.start.wrapping_sub(model.image_base);
                let mut bytes = read_padded(pe, rva, size as usize);

                for reloc in symbols.relocs_in(range.clone()) {
                    let Some(flags) = abs_flags(format, model.arch, reloc.size) else {
                        continue;
                    };
                    let offset = (reloc.addr - range.start) as usize;
                    let width = reloc.size as usize;
                    if offset + width > bytes.len() {
                        return Err(anyhow!(
                            "data split boundary intersects the relocation at {:#x}",
                            reloc.addr
                        ));
                    }
                    if let Some((name, addend)) =
                        resolve_split_data(symbols, owned_ranges, reloc.target)
                    {
                        bytes[offset..offset + width].fill(0);
                        pending.push(Pending {
                            sid: data_sid,
                            offset: offset as u64,
                            sym: name,
                            addend,
                            flags,
                        });
                    }
                }
                obj.append_section_data(data_sid, &bytes, 1);
            }

            let anchor = data_anchor(class, range.start);
            let anchor_id = obj.add_symbol(Symbol {
                name: anchor.as_bytes().to_vec(),
                value: 0,
                size: 0,
                kind: SymbolKind::Data,
                scope: SymbolScope::Dynamic,
                weak: false,
                section: SymbolSection::Section(data_sid),
                flags: SymbolFlags::None,
            });
            local.insert(anchor, anchor_id);

            if is_first_class_address(model, class, range.start) {
                let name = match class {
                    SegClass::Const => SYM_CONST_START,
                    SegClass::Data => SYM_DATA_START,
                    SegClass::Bss => SYM_BSS_START,
                    _ => unreachable!("only data classes are emitted here"),
                };
                let id = obj.add_symbol(Symbol {
                    name: name.as_bytes().to_vec(),
                    value: 0,
                    size: 0,
                    kind: SymbolKind::Data,
                    scope: SymbolScope::Dynamic,
                    weak: false,
                    section: SymbolSection::Section(data_sid),
                    flags: SymbolFlags::None,
                });
                local.insert(name.to_string(), id);
            }

            for (va, variable) in symbols.variables.range(range.clone()) {
                let scope = if variable.public {
                    SymbolScope::Dynamic
                } else {
                    SymbolScope::Compilation
                };
                let id = obj.add_symbol(Symbol {
                    name: sanitize_symbol_name(&variable.name),
                    value: va - range.start,
                    size: 0,
                    kind: SymbolKind::Data,
                    scope,
                    weak: false,
                    section: SymbolSection::Section(data_sid),
                    flags: SymbolFlags::None,
                });
                local.entry(variable.name.clone()).or_insert(id);
            }

            match class {
                SegClass::Const => stats.const_bytes += size,
                SegClass::Data => stats.data_bytes += size,
                SegClass::Bss => stats.bss_bytes += size,
                _ => unreachable!("only data classes are emitted here"),
            }
        }
    }

    for p in pending {
        let sym = resolve_symbol(&mut obj, &local, &mut undef, &p.sym, format);
        obj.add_relocation(
            p.sid,
            Relocation {
                offset: p.offset,
                symbol: sym,
                addend: p.addend,
                flags: p.flags,
            },
        )
        .with_context(|| format!("add reloc at {:#x}", p.offset))?;
        stats.relocations += 1;
    }

    stats.local_symbols = local.len();
    stats.undef_symbols = undef.len();

    let out = obj.write().context("serialize object")?;
    write_file(out_path, &out)?;
    Ok(stats)
}

// ---------------------------------------------------------------------------
// iced-x86 recovery (arch-dispatched, normalised to a single reloc shape)
// ---------------------------------------------------------------------------

struct Recovered {
    relocs: Vec<RecRel>,
    instructions: usize,
    unresolved_calls: usize,
    unresolved_rip: usize,
}

struct RecRel {
    offset: u64,
    target: String,
    addend: i64,
}

struct SplitResolver<'a> {
    symbols: &'a IdaSymbols,
    owned_ranges: &'a [OwnedDataRange],
}

impl delink_x86::recover::SymbolResolver for SplitResolver<'_> {
    fn resolve_code(&self, va: u64) -> Option<(String, i64)> {
        self.symbols.resolve_code(va)
    }

    fn resolve_data(&self, va: u64) -> Option<(String, i64)> {
        resolve_split_data(self.symbols, self.owned_ranges, va)
    }
}

impl delink_x86_64::recover::SymbolResolver for SplitResolver<'_> {
    fn resolve_code(&self, va: u64) -> Option<(String, i64)> {
        self.symbols.resolve_code(va)
    }

    fn resolve_data(&self, va: u64) -> Option<(String, i64)> {
        resolve_split_data(self.symbols, self.owned_ranges, va)
    }
}

fn recover(
    model: &IdaModel,
    bytes: &[u8],
    va: u64,
    size: u64,
    symbols: &IdaSymbols,
    owned_ranges: &[OwnedDataRange],
) -> Result<Recovered> {
    let resolver = SplitResolver {
        symbols,
        owned_ranges,
    };
    match model.arch {
        IdaArch::X86 => {
            let r = delink_x86::recover(bytes, va, size, &resolver)?;
            Ok(Recovered {
                relocs: r
                    .relocs
                    .into_iter()
                    .map(|x| RecRel {
                        offset: x.offset,
                        target: x.target,
                        addend: x.addend,
                    })
                    .collect(),
                instructions: r.diag.instructions,
                unresolved_calls: r.diag.calls_unresolved,
                unresolved_rip: r.diag.rip_refs_unresolved,
            })
        }
        IdaArch::X86_64 => {
            let r = delink_x86_64::recover(bytes, va, size, &resolver)?;
            Ok(Recovered {
                relocs: r
                    .relocs
                    .into_iter()
                    .map(|x| RecRel {
                        offset: x.offset,
                        target: x.target,
                        addend: x.addend,
                    })
                    .collect(),
                instructions: r.diag.instructions,
                unresolved_calls: r.diag.calls_unresolved,
                unresolved_rip: r.diag.rip_refs_unresolved,
            })
        }
        IdaArch::Other => Ok(Recovered {
            relocs: vec![],
            instructions: 0,
            unresolved_calls: 0,
            unresolved_rip: 0,
        }),
    }
}

// ---------------------------------------------------------------------------
// Relocation flag mapping
// ---------------------------------------------------------------------------

fn rel32_flags(fmt: OutputFormat, arch: IdaArch) -> RelocationFlags {
    match (fmt, arch) {
        (OutputFormat::Coff, IdaArch::X86_64) => RelocationFlags::Coff {
            typ: object::pe::IMAGE_REL_AMD64_REL32,
        },
        (OutputFormat::Coff, _) => RelocationFlags::Coff {
            typ: object::pe::IMAGE_REL_I386_REL32,
        },
        (OutputFormat::Elf, IdaArch::X86_64) => RelocationFlags::Elf {
            r_type: object::elf::R_X86_64_PC32,
        },
        (OutputFormat::Elf, _) => RelocationFlags::Elf {
            r_type: object::elf::R_386_PC32,
        },
    }
}

fn abs_flags(fmt: OutputFormat, arch: IdaArch, size: u32) -> Option<RelocationFlags> {
    Some(match (fmt, arch, size) {
        (OutputFormat::Coff, IdaArch::X86_64, 8) => RelocationFlags::Coff {
            typ: object::pe::IMAGE_REL_AMD64_ADDR64,
        },
        (OutputFormat::Coff, IdaArch::X86_64, 4) => RelocationFlags::Coff {
            typ: object::pe::IMAGE_REL_AMD64_ADDR32,
        },
        (OutputFormat::Coff, IdaArch::X86, 4) => RelocationFlags::Coff {
            typ: object::pe::IMAGE_REL_I386_DIR32,
        },
        (OutputFormat::Elf, IdaArch::X86_64, 8) => RelocationFlags::Elf {
            r_type: object::elf::R_X86_64_64,
        },
        (OutputFormat::Elf, IdaArch::X86_64, 4) => RelocationFlags::Elf {
            r_type: object::elf::R_X86_64_32,
        },
        (OutputFormat::Elf, IdaArch::X86, 4) => RelocationFlags::Elf {
            r_type: object::elf::R_386_32,
        },
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn data_anchor(class: SegClass, start: u64) -> String {
    let section = match class {
        SegClass::Const => "rdata",
        SegClass::Data => "data",
        SegClass::Bss => "bss",
        _ => unreachable!("data anchors only support data classes"),
    };
    format!("__delink_ida_{section}_{start:016x}")
}

fn owned_data_ranges(groups: &IdaproJson) -> Vec<OwnedDataRange> {
    groups
        .values()
        .flat_map(|group| {
            group
                .rdata
                .iter()
                .map(|range| (SegClass::Const, *range))
                .chain(group.data.iter().map(|range| (SegClass::Data, *range)))
                .chain(group.bss.iter().map(|range| (SegClass::Bss, *range)))
        })
        .map(|(class, configured)| {
            let range = configured.range();
            OwnedDataRange {
                anchor: data_anchor(class, range.start),
                range,
            }
        })
        .collect()
}

fn resolve_split_data(
    symbols: &IdaSymbols,
    owned_ranges: &[OwnedDataRange],
    va: u64,
) -> Option<(String, i64)> {
    // Preserve exact exported names (including synthetic names generated for
    // anonymous relocation targets) before mapping interior addresses to an
    // object's range anchor.
    if let Some(named) = symbols.resolve_exact(va) {
        return Some(named);
    }
    if let Some(owned) = owned_ranges.iter().find(|owned| owned.range.contains(&va)) {
        return Some((owned.anchor.clone(), (va - owned.range.start) as i64));
    }
    symbols.resolve_data(va)
}

fn validate_data_ranges(model: &IdaModel, groups: &IdaproJson) -> Result<()> {
    let mut all: Vec<(u64, u64, &str, &'static str)> = Vec::new();
    for (file, group) in groups.iter() {
        for (field, class, ranges) in [
            ("rdata", SegClass::Const, group.rdata.as_slice()),
            ("data", SegClass::Data, group.data.as_slice()),
            ("bss", SegClass::Bss, group.bss.as_slice()),
        ] {
            for configured in ranges {
                let range = configured.range();
                if range.start >= range.end {
                    return Err(anyhow!(
                        "idapro {file:?} {field} range [{:#x}, {:#x}) is empty or reversed",
                        range.start,
                        range.end
                    ));
                }
                let contained = model.sections.iter().any(|section| {
                    (if class == SegClass::Bss {
                        // IDA may merge the zero-initialized tail into its
                        // writable DATA segment. `bss` is therefore a logical
                        // output classification, not a requirement that the
                        // source segment already be named/classified BSS.
                        matches!(section.class, SegClass::Bss | SegClass::Data)
                    } else {
                        section.class == class
                    }) && section.start <= range.start
                        && range.end <= section.end
                });
                if !contained {
                    return Err(anyhow!(
                        "idapro {file:?} {field} range [{:#x}, {:#x}) is not contained in a compatible section",
                        range.start,
                        range.end
                    ));
                }
                all.push((range.start, range.end, file, field));
            }
        }
    }
    all.sort_by_key(|range| range.0);
    for pair in all.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(anyhow!(
                "overlapping idapro data ranges: {:?} {} [{:#x}, {:#x}) and {:?} {} [{:#x}, {:#x})",
                pair[0].2,
                pair[0].3,
                pair[0].0,
                pair[0].1,
                pair[1].2,
                pair[1].3,
                pair[1].0,
                pair[1].1
            ));
        }
    }
    Ok(())
}

/// Expand the optional whole-function ranges in an `idapro.json` grouping to
/// the explicit function starts consumed by the object emitter.
///
/// Ranges use the same half-open convention as data ranges. A range must be
/// contained in a code section and may only contain complete functions: a
/// range whose boundary falls inside a function is rejected. This makes the
/// selection deterministic even when there are gaps between IDA functions.
fn expand_function_ranges(model: &IdaModel, groups: &IdaproJson) -> Result<IdaproJson> {
    // Keep the legacy explicit-address path unchanged when no range syntax is
    // used. In particular, malformed explicit groups continue to be reported
    // as per-object emission failures rather than changing the split API's
    // error timing.
    if !groups
        .values()
        .any(|group| !group.function_ranges.is_empty())
    {
        return Ok(groups.clone());
    }

    let mut functions: Vec<&Function> = model.functions.iter().filter(|f| f.size() > 0).collect();
    functions.sort_by_key(|f| f.start);

    let mut all_ranges: Vec<(u64, u64, &str)> = Vec::new();
    for (file, group) in groups {
        for configured in &group.function_ranges {
            let range = configured.range();
            if range.start >= range.end {
                return Err(anyhow!(
                    "idapro {file:?} function range [{:#x}, {:#x}) is empty or reversed",
                    range.start,
                    range.end
                ));
            }
            if !model.sections.iter().any(|section| {
                section.class == SegClass::Code
                    && section.start <= range.start
                    && range.end <= section.end
            }) {
                return Err(anyhow!(
                    "idapro {file:?} function range [{:#x}, {:#x}) is not contained in a code section",
                    range.start,
                    range.end
                ));
            }
            all_ranges.push((range.start, range.end, file));
        }
    }
    all_ranges.sort_by_key(|range| range.0);
    for pair in all_ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(anyhow!(
                "overlapping idapro function ranges: {:?} [{:#x}, {:#x}) and {:?} [{:#x}, {:#x})",
                pair[0].2,
                pair[0].0,
                pair[0].1,
                pair[1].2,
                pair[1].0,
                pair[1].1
            ));
        }
    }

    let by_start: HashMap<u64, &Function> = functions.iter().map(|f| (f.start, *f)).collect();
    let mut owners: HashMap<u64, &str> = HashMap::new();
    let mut expanded = groups.clone();

    for (file, group) in groups {
        let mut starts = group.functions.clone();
        for &address in &group.functions {
            let function = by_start.get(&address).ok_or_else(|| {
                anyhow!("idapro address {address:#x} is not a function start in delink.json")
            })?;
            if function.size() == 0 {
                return Err(anyhow!(
                    "function '{}' at {address:#x} has zero size in delink.json",
                    function.name
                ));
            }
            if let Some(previous) = owners.insert(address, file.as_str()) {
                let message = if previous == file.as_str() {
                    "assigned more than once in the same idapro group"
                } else {
                    "assigned to more than one idapro group"
                };
                return Err(anyhow!("function address {address:#x} {message}"));
            }
        }

        for configured in &group.function_ranges {
            let range = configured.range();
            let first = functions.partition_point(|f| f.start < range.start);
            if first > 0 {
                let preceding = functions[first - 1];
                if preceding.end > range.start {
                    return Err(anyhow!(
                        "idapro {file:?} function range [{:#x}, {:#x}) cuts through function '{}' [{:#x}, {:#x})",
                        range.start,
                        range.end,
                        preceding.name,
                        preceding.start,
                        preceding.end
                    ));
                }
            }
            let mut selected = 0usize;
            for function in functions.iter().skip(first) {
                if function.start >= range.end {
                    break;
                }
                if function.start < range.start || function.end > range.end {
                    return Err(anyhow!(
                        "idapro {file:?} function range [{:#x}, {:#x}) cuts through function '{}' [{:#x}, {:#x})",
                        range.start,
                        range.end,
                        function.name,
                        function.start,
                        function.end
                    ));
                }
                if let Some(previous) = owners.insert(function.start, file.as_str()) {
                    let message = if previous == file.as_str() {
                        "assigned more than once in the same idapro group"
                    } else {
                        "assigned to more than one idapro group"
                    };
                    return Err(anyhow!("function address {:#x} {message}", function.start));
                }
                starts.push(function.start);
                selected += 1;
            }
            if selected == 0 {
                return Err(anyhow!(
                    "idapro {file:?} function range [{:#x}, {:#x}) contains no functions",
                    range.start,
                    range.end
                ));
            }
        }

        expanded
            .get_mut(file)
            .expect("group copied from the same map")
            .functions = starts;
    }
    Ok(expanded)
}

fn subtract_ranges(whole: Range<u64>, excluded: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut cuts: Vec<Range<u64>> = excluded
        .iter()
        .filter_map(|range| {
            let start = whole.start.max(range.start);
            let end = whole.end.min(range.end);
            (start < end).then_some(start..end)
        })
        .collect();
    cuts.sort_by_key(|range| range.start);

    let mut result = Vec::new();
    let mut cursor = whole.start;
    for cut in cuts {
        if cursor < cut.start {
            result.push(cursor..cut.start);
        }
        cursor = cursor.max(cut.end);
    }
    if cursor < whole.end {
        result.push(cursor..whole.end);
    }
    result
}

fn is_first_class_address(model: &IdaModel, class: SegClass, address: u64) -> bool {
    model
        .sections
        .iter()
        .find(|section| section.class == class)
        .is_some_and(|section| section.start == address)
}

fn obj_arch(model: &IdaModel) -> (Architecture, Endianness) {
    let arch = match model.arch {
        IdaArch::X86 => Architecture::I386,
        IdaArch::X86_64 => Architecture::X86_64,
        IdaArch::Other => Architecture::Unknown,
    };
    let endian = if model.little_endian {
        Endianness::Little
    } else {
        Endianness::Big
    };
    (arch, endian)
}

fn text_name(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Coff => ".text",
        OutputFormat::Elf => ".text",
    }
}
fn data_name(_fmt: OutputFormat) -> &'static str {
    ".data"
}
fn const_name(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Coff => ".rdata",
        OutputFormat::Elf => ".rodata",
    }
}
fn bss_name(_fmt: OutputFormat) -> &'static str {
    ".bss"
}

fn resolve_symbol(
    obj: &mut Object,
    local: &HashMap<String, SymbolId>,
    undef: &mut HashMap<String, SymbolId>,
    name: &str,
    fmt: OutputFormat,
) -> SymbolId {
    if let Some(id) = local.get(name) {
        return *id;
    }
    resolve_or_add_undef(obj, undef, name, fmt)
}

fn resolve_or_add_undef(
    obj: &mut Object,
    undef: &mut HashMap<String, SymbolId>,
    name: &str,
    fmt: OutputFormat,
) -> SymbolId {
    if let Some(id) = undef.get(name) {
        return *id;
    }
    // COFF doesn't support SymbolKind::Unknown; use Data for undefined externs.
    let kind = match fmt {
        OutputFormat::Coff => SymbolKind::Data,
        OutputFormat::Elf => SymbolKind::Unknown,
    };
    let id = obj.add_symbol(Symbol {
        name: sanitize_symbol_name(name),
        value: 0,
        size: 0,
        kind,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    undef.insert(name.to_string(), id);
    id
}

fn sanitize_symbol_name(name: &str) -> Vec<u8> {
    if name.is_empty() {
        return b"<invalid>".to_vec();
    }
    name.as_bytes().to_vec()
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::idapro_json::DataRange;
    use delink_pe::{PeArch, PeSection};
    use object::{Object as _, ObjectSection as _};

    fn test_model() -> (IdaModel, PeImage, IdaSymbols) {
        let model = IdaModel {
            arch: IdaArch::X86_64,
            procname: "metapc".into(),
            bits: 64,
            little_endian: true,
            image_base: 0x1000,
            filetype: "PE".into(),
            input_file: "test.exe".into(),
            sections: vec![
                crate::Section {
                    name: ".rdata".into(),
                    start: 0x1100,
                    end: 0x1110,
                    read: true,
                    write: false,
                    exec: false,
                    class: SegClass::Const,
                },
                crate::Section {
                    name: ".data".into(),
                    start: 0x1200,
                    end: 0x1220,
                    read: true,
                    write: true,
                    exec: false,
                    class: SegClass::Data,
                },
            ],
            functions: vec![],
            names: vec![],
            relocations: vec![],
        };
        let pe = PeImage {
            arch: PeArch::X86_64,
            image_base: 0x1000,
            sections: vec![
                PeSection {
                    name: ".rdata".into(),
                    rva: 0x100,
                    va: 0x1100,
                    virtual_size: 0x10,
                    data: (0..0x10).collect(),
                    characteristics: 0,
                },
                PeSection {
                    name: ".data".into(),
                    rva: 0x200,
                    va: 0x1200,
                    virtual_size: 0x20,
                    data: (0x20..0x40).collect(),
                    characteristics: 0,
                },
            ],
            base_relocations: vec![],
        };
        let symbols = IdaSymbols::build(&model, &[]);
        (model, pe, symbols)
    }

    fn section_data(path: &Path, name: &str) -> Vec<u8> {
        let bytes = std::fs::read(path).unwrap();
        let file = object::File::parse(bytes.as_slice()).unwrap();
        file.sections()
            .filter(|section| section.name().ok() == Some(name))
            .flat_map(|section| section.data().unwrap().to_vec())
            .collect()
    }

    fn section_size(path: &Path, name: &str) -> u64 {
        let bytes = std::fs::read(path).unwrap();
        let file = object::File::parse(bytes.as_slice()).unwrap();
        file.sections()
            .find(|section| section.name().ok() == Some(name))
            .map(|section| section.size())
            .unwrap_or(0)
    }

    #[test]
    fn assigned_ranges_move_out_of_shared_data() {
        let (model, pe, symbols) = test_model();
        let mut groups = BTreeMap::new();
        groups.insert(
            "owner.obj".to_string(),
            ObjectGroup {
                functions: vec![],
                function_ranges: vec![],
                rdata: vec![DataRange([0x1104, 0x1108])],
                data: vec![DataRange([0x1208, 0x1210])],
                bss: vec![],
            },
        );

        let dir = std::env::temp_dir().join(format!(
            "delink-ida-data-split-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let outcomes =
            split_by_groups(&model, &pe, &symbols, &groups, &dir, OutputFormat::Coff).unwrap();
        assert_eq!(outcomes.len(), 1);
        let stats = outcomes[0].result.as_ref().unwrap();
        assert_eq!(stats.const_bytes, 4);
        assert_eq!(stats.data_bytes, 8);
        assert_eq!(
            section_data(&dir.join("owner.obj"), ".rdata"),
            vec![4, 5, 6, 7]
        );
        assert_eq!(
            section_data(&dir.join("owner.obj"), ".data"),
            (0x28..0x30).collect::<Vec<_>>()
        );

        let shared = dir.join("__shared_data.obj");
        let shared_stats =
            emit_shared_for_groups(&model, &pe, &symbols, &groups, &shared, OutputFormat::Coff)
                .unwrap();
        assert_eq!(shared_stats.const_bytes, 12);
        assert_eq!(shared_stats.data_bytes, 24);
        assert_eq!(
            section_data(&shared, ".rdata"),
            vec![0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]
        );
        assert_eq!(
            section_data(&shared, ".data"),
            (0x20..0x28).chain(0x30..0x40).collect::<Vec<_>>()
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bss_range_from_data_becomes_zero_filled_bss() {
        let (model, pe, symbols) = test_model();
        let groups = BTreeMap::from([(
            "owner.obj".to_string(),
            ObjectGroup {
                bss: vec![DataRange([0x1210, 0x1218])],
                ..Default::default()
            },
        )]);
        let dir = std::env::temp_dir().join(format!(
            "delink-ida-bss-split-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let outcomes =
            split_by_groups(&model, &pe, &symbols, &groups, &dir, OutputFormat::Coff).unwrap();
        assert_eq!(outcomes[0].result.as_ref().unwrap().bss_bytes, 8);
        assert_eq!(section_size(&dir.join("owner.obj"), ".bss"), 8);
        assert!(section_data(&dir.join("owner.obj"), ".bss").is_empty());

        let shared = dir.join("__shared_data.obj");
        let stats =
            emit_shared_for_groups(&model, &pe, &symbols, &groups, &shared, OutputFormat::Coff)
                .unwrap();
        assert_eq!(stats.data_bytes, 24);
        assert_eq!(
            section_data(&shared, ".data"),
            (0x20..0x30).chain(0x38..0x40).collect::<Vec<_>>()
        );

        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn rejects_overlapping_ranges() {
        let (model, _, _) = test_model();
        let groups = BTreeMap::from([
            (
                "one.obj".into(),
                ObjectGroup {
                    rdata: vec![DataRange([0x1100, 0x1108])],
                    ..Default::default()
                },
            ),
            (
                "two.obj".into(),
                ObjectGroup {
                    rdata: vec![DataRange([0x1104, 0x110c])],
                    ..Default::default()
                },
            ),
        ]);
        assert!(validate_data_ranges(&model, &groups)
            .unwrap_err()
            .to_string()
            .contains("overlapping"));
    }

    #[test]
    fn function_ranges_select_complete_functions() {
        let model = IdaModel {
            arch: IdaArch::X86_64,
            procname: "metapc".into(),
            bits: 64,
            little_endian: true,
            image_base: 0x1000,
            filetype: "PE".into(),
            input_file: "test.exe".into(),
            sections: vec![crate::Section {
                name: ".text".into(),
                start: 0x1000,
                end: 0x1020,
                read: true,
                write: false,
                exec: true,
                class: SegClass::Code,
            }],
            functions: vec![
                Function {
                    start: 0x1000,
                    end: 0x1004,
                    name: "one".into(),
                    thunk: false,
                    lib: false,
                    is_static: false,
                    public: false,
                },
                Function {
                    start: 0x1008,
                    end: 0x1010,
                    name: "two".into(),
                    thunk: false,
                    lib: false,
                    is_static: false,
                    public: false,
                },
            ],
            names: vec![],
            relocations: vec![],
        };
        let groups = BTreeMap::from([(
            "all.obj".into(),
            ObjectGroup {
                function_ranges: vec![DataRange([0x1000, 0x1010])],
                ..Default::default()
            },
        )]);
        let expanded = expand_function_ranges(&model, &groups).unwrap();
        assert_eq!(expanded["all.obj"].functions, vec![0x1000, 0x1008]);
    }

    #[test]
    fn function_ranges_reject_partial_function() {
        let model = IdaModel {
            arch: IdaArch::X86_64,
            procname: "metapc".into(),
            bits: 64,
            little_endian: true,
            image_base: 0x1000,
            filetype: "PE".into(),
            input_file: "test.exe".into(),
            sections: vec![crate::Section {
                name: ".text".into(),
                start: 0x1000,
                end: 0x1010,
                read: true,
                write: false,
                exec: true,
                class: SegClass::Code,
            }],
            functions: vec![Function {
                start: 0x1000,
                end: 0x1008,
                name: "one".into(),
                thunk: false,
                lib: false,
                is_static: false,
                public: false,
            }],
            names: vec![],
            relocations: vec![],
        };
        let groups = BTreeMap::from([(
            "partial.obj".into(),
            ObjectGroup {
                function_ranges: vec![DataRange([0x1004, 0x1010])],
                ..Default::default()
            },
        )]);
        let error = expand_function_ranges(&model, &groups).unwrap_err();
        assert!(error.to_string().contains("cuts through function"));
    }

    #[test]
    fn exact_data_names_take_precedence_over_range_anchors() {
        let (mut model, _, _) = test_model();
        model.names.push(crate::Name {
            addr: 0x1208,
            name: "renamable_data".into(),
            public: false,
            weak: false,
            is_func: false,
        });
        let symbols = IdaSymbols::build(&model, &[]);
        let owned = [OwnedDataRange {
            range: 0x1200..0x1210,
            anchor: "__delink_ida_data_0000000000001200".into(),
        }];
        assert_eq!(
            resolve_split_data(&symbols, &owned, 0x1208),
            Some(("renamable_data".into(), 0))
        );
    }
}
