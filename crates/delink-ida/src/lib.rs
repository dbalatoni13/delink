//! delink IDA importer.
//!
//! Consumes the JSON produced by [`ida_export.py`](../ida_export.py) (run inside
//! IDA 9.x) and splits the analysed binary into relocatable objects.
//!
//! The JSON carries the architecture, every segment's bytes, every function
//! (boundaries + flags), switch jump tables, the full address → name map, and
//! IDA's fixup table.
//! For x86 / x86-64 targets the emitter disassembles each function with
//! iced-x86 (via [`delink_x86`] / [`delink_x86_64`]) to recover instruction
//! relocations — rel32 calls/jumps and RIP-relative references — resolving each
//! target address through the name map to build the correct label.  Absolute
//! pointers (in code or data) come from IDA's fixup table. Switch metadata is
//! kept separate from function bounds so trailing tables are emitted as data
//! even when IDA ends the function at its final instruction.
//!
//! Like the Mach-O `symtab.json` flow, the split is driven by an editable
//! `idapro.json` mapping each output object filename to explicit function start
//! addresses, optional whole-function ranges, and optional `.rdata` / `.data`
//! / logical `.bss`
//! address ranges. Names and bounds come from the exported model; named
//! functions and data are emitted with external linkage so separate objects
//! can reference them even when IDA did not mark them public. See
//! [`idapro_json`].
//!
//! The original input can be a PE image or an original Xbox XBE. XBE section
//! bytes are loaded from the XBE section table, while absolute relocations come
//! from the IDA export because XBE images do not carry a PE `.reloc` table.

pub mod emit;
pub mod idapro_json;
pub mod resolver;

use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

pub use delink_pe::{BaseRelocKind, PeImage};
pub use resolver::IdaSymbols;

/// Target architecture, as far as the importer cares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdaArch {
    X86,
    X86_64,
    /// Anything else — emitted as raw bytes with fixup-table relocations only
    /// (no iced-x86 instruction recovery).
    Other,
}

impl IdaArch {
    fn from_meta(arch: &str) -> Self {
        match arch {
            "x86" => IdaArch::X86,
            "x86_64" => IdaArch::X86_64,
            _ => IdaArch::Other,
        }
    }
}

/// Segment class, derived from IDA's segment type + permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegClass {
    Code,
    Data,
    Const,
    Bss,
    Xtrn,
    Other,
}

impl SegClass {
    fn from_str(s: &str) -> Self {
        match s {
            "CODE" => SegClass::Code,
            "DATA" => SegClass::Data,
            "CONST" => SegClass::Const,
            "BSS" => SegClass::Bss,
            "XTRN" => SegClass::Xtrn,
            _ => SegClass::Other,
        }
    }
}

/// A segment's metadata. The bytes live in the original input binary, not in
/// the export — read them through [`PeImage`] using image-base-relative
/// addresses (PE RVAs or the equivalent XBE mapping).
#[derive(Debug, Clone)]
pub struct Section {
    pub name: String,
    pub start: u64,
    pub end: u64,
    pub read: bool,
    pub write: bool,
    pub exec: bool,
    pub class: SegClass,
}

impl Section {
    pub fn size(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
    pub fn contains(&self, va: u64) -> bool {
        va >= self.start && va < self.end
    }
}

/// A function discovered by IDA.
#[derive(Debug, Clone)]
pub struct Function {
    pub start: u64,
    pub end: u64,
    pub name: String,
    pub thunk: bool,
    pub lib: bool,
    pub is_static: bool,
    pub public: bool,
}

impl Function {
    pub fn size(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// A named address.
#[derive(Debug, Clone)]
pub struct Name {
    pub addr: u64,
    /// Declared byte length of a data symbol (zero for code labels/legacy exports).
    pub size: u64,
    pub name: String,
    pub public: bool,
    pub weak: bool,
    pub is_func: bool,
}

/// One entry of IDA's fixup table (an absolute address relocation).
#[derive(Debug, Clone)]
pub struct Reloc {
    pub addr: u64,
    pub kind: String,
    /// Width in bytes (4 or 8).
    pub size: u32,
    /// The address the relocation points at.
    pub target: u64,
}

#[derive(Debug, Clone)]
pub struct JumpTableEntry {
    pub addr: u64,
    pub target: u64,
}

#[derive(Debug, Clone)]
pub struct JumpTable {
    pub owner: u64,
    pub dispatch: u64,
    pub dispatch_addr: Option<u64>,
    /// Additional encoded fields when more than one dispatch uses this table.
    pub dispatch_addrs: Vec<u64>,
    pub start: u64,
    pub entry_size: u32,
    pub entries: Vec<JumpTableEntry>,
    pub name: String,
}

impl JumpTable {
    pub fn end(&self) -> u64 {
        self.start + self.entry_size as u64 * self.entries.len() as u64
    }

    pub fn dispatch_fields(&self) -> impl Iterator<Item = u64> + '_ {
        self.dispatch_addr
            .into_iter()
            .chain(self.dispatch_addrs.iter().copied())
    }
}

/// The fully decoded model.
pub struct IdaModel {
    pub arch: IdaArch,
    pub procname: String,
    pub bits: u32,
    pub little_endian: bool,
    pub image_base: u64,
    pub filetype: String,
    pub input_file: String,
    pub sections: Vec<Section>,
    pub functions: Vec<Function>,
    pub names: Vec<Name>,
    pub relocations: Vec<Reloc>,
    pub jump_tables: Vec<JumpTable>,
}

impl IdaModel {
    pub fn section_for(&self, va: u64) -> Option<&Section> {
        self.sections.iter().find(|s| s.contains(va))
    }
}

// ---------------------------------------------------------------------------
// Raw (serde) schema — kept private; `load` converts to `IdaModel`.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawModel {
    #[allow(dead_code)]
    delink_ida_version: u32,
    meta: RawMeta,
    segments: Vec<RawSegment>,
    functions: Vec<RawFunction>,
    names: Vec<RawName>,
    relocations: Vec<RawReloc>,
    #[serde(default)]
    jump_tables: Vec<RawJumpTable>,
}

#[derive(Deserialize)]
struct RawMeta {
    arch: String,
    procname: String,
    bits: u32,
    endian: String,
    image_base: u64,
    #[allow(dead_code)]
    min_ea: u64,
    #[allow(dead_code)]
    max_ea: u64,
    filetype: String,
    input_file: String,
}

#[derive(Deserialize)]
struct RawSegment {
    name: String,
    start: u64,
    end: u64,
    perm_r: bool,
    perm_w: bool,
    perm_x: bool,
    class: String,
    #[allow(dead_code)]
    #[serde(default)]
    bitness: u32,
}

#[derive(Deserialize)]
struct RawFunction {
    start: u64,
    #[serde(default)]
    end: Option<u64>,
    #[serde(default)]
    size: Option<serde_json::Value>,
    name: String,
    thunk: bool,
    lib: bool,
    #[serde(rename = "static")]
    is_static: bool,
    public: bool,
    #[allow(dead_code)]
    thunk_target: Option<u64>,
}

#[derive(Deserialize)]
struct RawName {
    addr: u64,
    #[serde(default)]
    size: Option<serde_json::Value>,
    name: String,
    public: bool,
    weak: bool,
    is_func: bool,
}

#[derive(Deserialize)]
struct RawReloc {
    addr: u64,
    #[serde(rename = "type")]
    kind: String,
    size: u32,
    target: u64,
}

#[derive(Deserialize)]
struct RawJumpTableEntry {
    addr: u64,
    target: u64,
}

#[derive(Deserialize)]
struct RawJumpTable {
    owner: u64,
    dispatch: u64,
    dispatch_addr: Option<u64>,
    start: u64,
    entry_size: u32,
    entries: Vec<RawJumpTableEntry>,
    name: String,
}

fn parse_size(value: &serde_json::Value) -> Result<u64> {
    match value {
        serde_json::Value::String(text) => {
            let hex = text
                .strip_prefix("0x")
                .or_else(|| text.strip_prefix("0X"))
                .context("size must be a hexadecimal string such as 0x4")?;
            u64::from_str_radix(hex, 16).context("invalid hexadecimal size")
        }
        serde_json::Value::Number(number) => number.as_u64().context("size must be nonnegative"),
        _ => bail!("size must be a hexadecimal string or integer"),
    }
}

fn validate_data_symbols(sections: &[Section], names: &[Name]) -> Result<()> {
    let mut data: Vec<&Name> = names
        .iter()
        .filter(|name| {
            !name.is_func
                && sections.iter().any(|section| {
                    section.contains(name.addr)
                        && matches!(
                            section.class,
                            SegClass::Data | SegClass::Const | SegClass::Bss
                        )
                })
        })
        .collect();
    data.sort_by_key(|name| name.addr);
    for (index, name) in data.iter().enumerate() {
        let end = name
            .addr
            .checked_add(name.size)
            .context("data size overflows address space")?;
        let section = sections
            .iter()
            .find(|section| section.contains(name.addr))
            .unwrap();
        ensure!(
            end <= section.end,
            "data symbol {} at {:#x} extends beyond section {:#x}",
            name.name,
            name.addr,
            section.end
        );
        if let Some(next) = data.get(index + 1) {
            ensure!(
                end <= next.addr,
                "data symbols {} at {:#x} and {} at {:#x} overlap",
                name.name,
                name.addr,
                next.name,
                next.addr
            );
        }
    }
    Ok(())
}

/// Load and decode an exported `*.delink.json` file.
pub fn load(path: &Path) -> Result<IdaModel> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: RawModel =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;

    let sections: Vec<Section> = raw
        .segments
        .into_iter()
        .map(|s| Section {
            name: s.name,
            start: s.start,
            end: s.end,
            read: s.perm_r,
            write: s.perm_w,
            exec: s.perm_x,
            class: SegClass::from_str(&s.class),
        })
        .collect();

    let mut functions: Vec<Function> = raw
        .functions
        .into_iter()
        .map(|f| {
            let end = match (f.size.as_ref(), f.end) {
                (Some(size), None) => f
                    .start
                    .checked_add(parse_size(size)?)
                    .context("function size overflows address space")?,
                (None, Some(end)) => end,
                _ => anyhow::bail!("function {} needs exactly one of size or end", f.name),
            };
            anyhow::ensure!(end > f.start, "function {} has invalid size", f.name);
            Ok(Function {
                start: f.start,
                end,
                name: f.name,
                thunk: f.thunk,
                lib: f.lib,
                is_static: f.is_static,
                public: f.public,
            })
        })
        .collect::<Result<_>>()?;
    functions.sort_by_key(|function| function.start);

    let names: Vec<Name> = raw
        .names
        .into_iter()
        .map(|n| {
            let size = n.size.as_ref().map(parse_size).transpose()?.unwrap_or(0);
            if n.size.is_some()
                && !n.is_func
                && sections.iter().any(|section| {
                    section.contains(n.addr)
                        && matches!(
                            section.class,
                            SegClass::Data | SegClass::Const | SegClass::Bss
                        )
                })
            {
                ensure!(
                    size > 0,
                    "data symbol {} at {:#x} has zero size",
                    n.name,
                    n.addr
                );
            }
            Ok(Name {
                addr: n.addr,
                size,
                name: n.name,
                public: n.public,
                weak: n.weak,
                is_func: n.is_func,
            })
        })
        .collect::<Result<_>>()?;
    validate_data_symbols(&sections, &names)?;

    let relocations = raw
        .relocations
        .into_iter()
        .map(|r| Reloc {
            addr: r.addr,
            kind: r.kind,
            size: r.size,
            target: r.target,
        })
        .collect();

    let jump_tables = raw
        .jump_tables
        .into_iter()
        .map(|table| JumpTable {
            owner: table.owner,
            dispatch: table.dispatch,
            dispatch_addr: table.dispatch_addr,
            dispatch_addrs: Vec::new(),
            start: table.start,
            entry_size: table.entry_size,
            entries: table
                .entries
                .into_iter()
                .map(|entry| JumpTableEntry {
                    addr: entry.addr,
                    target: entry.target,
                })
                .collect(),
            name: table.name,
        })
        .collect();

    Ok(IdaModel {
        arch: IdaArch::from_meta(&raw.meta.arch),
        procname: raw.meta.procname,
        bits: raw.meta.bits,
        little_endian: raw.meta.endian != "big",
        image_base: raw.meta.image_base,
        filetype: raw.meta.filetype,
        input_file: raw.meta.input_file,
        sections,
        functions,
        names,
        relocations,
        jump_tables,
    })
}

/// Load the original input binary (PE or original Xbox XBE) whose bytes the
/// split will carve up.
pub fn load_binary(path: &Path) -> Result<PeImage> {
    let data = std::fs::read(path).with_context(|| format!("read binary {}", path.display()))?;
    delink_pe::load_pe_image(&data)
        .with_context(|| format!("parse PE/XBE image {}", path.display()))
}

/// The set of absolute-pointer relocations to apply, in IDA-VA space.
///
/// Combines two sources (deduplicated by address):
///   * the relocations IDA exported (its fixup table — the only source for EXEs,
///     whose images carry no `.reloc`), and
///   * the PE base-relocation table from the original binary (present in DLLs),
///     whose targets are read from the binary's own bytes.
///
/// XBE images do not have a PE base-relocation table, so their absolute
/// relocations are supplied by the first source (IDA's fixup export).
///
/// Both are translated into IDA's address space via the image-base delta so they
/// resolve against the same name map.
pub fn combined_relocations(model: &IdaModel, pe: &PeImage) -> Vec<Reloc> {
    use std::collections::BTreeMap;

    let mut by_addr: BTreeMap<u64, Reloc> = BTreeMap::new();

    // 1) IDA fixups and offset-typed operands. An offset operand can describe
    // a structure member displacement rather than an address; only recover an
    // absolute relocation when the encoded value agrees with its target.
    for r in &model.relocations {
        let Some(bytes) = pe.data_at_rva(r.addr.wrapping_sub(model.image_base), r.size as usize)
        else {
            continue;
        };
        let stored = match r.size {
            4 => u32::from_le_bytes(bytes.try_into().unwrap()) as u64,
            8 => u64::from_le_bytes(bytes.try_into().unwrap()),
            _ => continue,
        };
        let rebased = model
            .image_base
            .wrapping_add(stored.wrapping_sub(pe.image_base));
        if stored != r.target && rebased != r.target {
            continue;
        }
        by_addr.insert(r.addr, r.clone());
    }

    // 2) PE `.reloc` entries — translate RVA → IDA VA, read the target from the binary.
    let ida_base = model.image_base;
    let pe_base = pe.image_base;
    for br in &pe.base_relocations {
        let size: u32 = match br.kind {
            BaseRelocKind::Dir64 => 8,
            BaseRelocKind::HighLow => 4,
            BaseRelocKind::Other(_) => continue,
        };
        let rva = br.va.wrapping_sub(pe_base);
        let addr_ida = ida_base.wrapping_add(rva);
        if by_addr.contains_key(&addr_ida) {
            continue; // already covered by an IDA fixup
        }
        let Some(bytes) = pe.data_at_rva(rva, size as usize) else {
            continue;
        };
        let stored = match size {
            8 => u64::from_le_bytes(bytes.try_into().unwrap()),
            _ => u32::from_le_bytes(bytes.try_into().unwrap()) as u64,
        };
        // The stored value is an absolute VA at the binary's base; rebase to IDA.
        let target_ida = ida_base.wrapping_add(stored.wrapping_sub(pe_base));
        by_addr.insert(
            addr_ida,
            Reloc {
                addr: addr_ida,
                kind: "RELOC".to_string(),
                size,
                target: target_ida,
            },
        );
    }

    by_addr.into_values().collect()
}

#[cfg(test)]
mod size_tests {
    use super::*;

    fn section() -> Section {
        Section {
            name: ".rdata".into(),
            start: 0x1000,
            end: 0x1100,
            read: true,
            write: false,
            exec: false,
            class: SegClass::Const,
        }
    }

    fn name(addr: u64, size: u64, text: &str) -> Name {
        Name {
            addr,
            size,
            name: text.into(),
            public: false,
            weak: false,
            is_func: false,
        }
    }

    #[test]
    fn parses_hexadecimal_sizes() {
        assert_eq!(parse_size(&serde_json::json!("0xC")).unwrap(), 12);
        assert_eq!(parse_size(&serde_json::json!("0x10")).unwrap(), 16);
        assert!(parse_size(&serde_json::json!("12")).is_err());
    }

    #[test]
    fn rejects_overlapping_data_symbols() {
        let sections = [section()];
        let names = [name(0x1004, 12, "kZero"), name(0x1008, 4, "dword")];
        let error = validate_data_symbols(&sections, &names).unwrap_err();
        assert!(error.to_string().contains("overlap"));
        let names = [name(0x1004, 12, "kZero"), name(0x1010, 4, "next")];
        validate_data_symbols(&sections, &names).unwrap();
    }

    #[test]
    fn rejects_data_that_extends_past_section() {
        let error =
            validate_data_symbols(&[section()], &[name(0x10fc, 8, "past_end")]).unwrap_err();
        assert!(error.to_string().contains("beyond section"));
    }
}
