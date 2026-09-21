//! PE (.exe/.dll) + PDB loader for `delink`.
//!
//! Supports both 64-bit (PE32+, machine AMD64) and 32-bit (PE32, machine I386)
//! PE executables together with their PDB files.

use anyhow::{anyhow, Result};
use std::collections::HashMap;

pub mod cu;
pub mod emit;
pub mod mangle;
pub mod symbols;

pub use cu::{PeCompilationUnit, PeContrib, PeCuIndex, PeFunction, PeVariable};
pub use symbols::PeGlobalSymbols;

/// Target architecture of the loaded PE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeArch {
    /// AMD64 / x86-64 (machine = 0x8664, magic = 0x020B PE32+).
    X86_64,
    /// IA-32 / x86 (machine = 0x014C, magic = 0x010B PE32).
    X86,
}

/// A single PE section with its raw bytes and virtual-address metadata.
#[derive(Debug, Clone)]
pub struct PeSection {
    pub name: String,
    /// RVA (relative virtual address from image base).
    pub rva: u64,
    /// Absolute virtual address = image_base + rva.
    pub va: u64,
    /// Virtual size (may be larger than raw data due to BSS padding).
    pub virtual_size: u64,
    /// Raw bytes from the file, zero-extended to `virtual_size` if needed.
    pub data: Vec<u8>,
    pub characteristics: u32,
}

impl PeSection {
    pub fn contains_rva(&self, rva: u64) -> bool {
        rva >= self.rva && rva < self.rva + self.virtual_size
    }

    pub fn contains_va(&self, va: u64) -> bool {
        va >= self.va && va < self.va + self.virtual_size
    }

    /// Borrow the bytes at `rva` (RVA relative to image base) for `len` bytes.
    pub fn data_at_rva(&self, rva: u64, len: usize) -> Option<&[u8]> {
        if rva < self.rva {
            return None;
        }
        let off = (rva - self.rva) as usize;
        let end = off.checked_add(len)?;
        if end > self.data.len() {
            return None;
        }
        Some(&self.data[off..end])
    }

    /// Borrow the bytes at an absolute VA for `len` bytes.
    pub fn data_at_va(&self, va: u64, len: usize) -> Option<&[u8]> {
        if va < self.va {
            return None;
        }
        let rva = va - self.va + self.rva;
        self.data_at_rva(rva, len)
    }
}

/// A base-relocation entry from the `.reloc` section.
#[derive(Debug, Clone)]
pub struct BaseReloc {
    pub va: u64,
    pub kind: BaseRelocKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseRelocKind {
    /// `IMAGE_REL_BASED_DIR64` (10) — 64-bit absolute pointer (PE32+).
    Dir64,
    /// `IMAGE_REL_BASED_HIGHLOW` (3) — 32-bit absolute pointer (PE32).
    HighLow,
    /// Other type we don't handle.
    Other(u8),
}

/// All parsed data from a PE + PDB pair.
pub struct PeContext {
    pub arch: PeArch,
    pub image_base: u64,
    pub sections: Vec<PeSection>,
    pub cu_index: PeCuIndex,
    pub symbols: PeGlobalSymbols,
    /// Base relocations from `.reloc` — one entry per embedded absolute pointer.
    pub base_relocations: Vec<BaseReloc>,
    /// IAT slot VA → `"__imp_funcname"` symbol name.
    pub imports: HashMap<u64, String>,
    /// Mangled names of procedures the PDB marks as declared inline
    /// (`S_FRAMEPROC` `fInlSpec`), sorted and de-duplicated.
    pub inlined_functions: Vec<String>,
}

impl PeContext {
    /// Find the section that contains `va`.
    pub fn section_for_va(&self, va: u64) -> Option<&PeSection> {
        self.sections.iter().find(|s| s.contains_va(va))
    }

    /// Bytes at an absolute VA, spanning across a single section.
    pub fn data_at_va(&self, va: u64, len: usize) -> Option<&[u8]> {
        self.section_for_va(va)?.data_at_va(va, len)
    }
}

/// Load a PE executable and its associated PDB file.
///
/// Accepts both 32-bit (PE32, machine I386) and 64-bit (PE32+, machine AMD64).
pub fn load_pe_and_pdb(exe_data: &[u8], pdb_data: &[u8]) -> Result<PeContext> {
    let (arch, image_base, sections) = parse_pe_sections(exe_data)?;
    let base_relocations = parse_base_relocations(&sections, image_base);
    let imports = parse_imports(exe_data, &sections, image_base, arch);

    let (cu_index, all_functions, all_variables, tls_variables, code_publics, inlined_functions) =
        cu::build_cu_index(pdb_data, image_base, &sections, arch)?;
    let symbols = symbols::PeGlobalSymbols::build(
        all_functions,
        all_variables,
        tls_variables,
        code_publics,
        &imports,
        &sections,
        image_base,
    );

    Ok(PeContext {
        arch,
        image_base,
        sections,
        cu_index,
        symbols,
        base_relocations,
        imports,
        inlined_functions,
    })
}

/// A PE image without debug info: just sections, image base, and the base
/// relocation table.
///
/// Lighter than [`PeContext`]; for callers (e.g. the IDA importer) that bring
/// their own function/symbol information and only need the original binary's
/// bytes and `.reloc` table.
pub struct PeImage {
    pub arch: PeArch,
    pub image_base: u64,
    pub sections: Vec<PeSection>,
    /// Base relocations from `.reloc` (empty for images without the table).
    pub base_relocations: Vec<BaseReloc>,
}

impl PeImage {
    pub fn section_for_rva(&self, rva: u64) -> Option<&PeSection> {
        self.sections.iter().find(|s| s.contains_rva(rva))
    }

    /// Bytes at `rva` for `len` bytes (sections are zero-extended to virtual size).
    pub fn data_at_rva(&self, rva: u64, len: usize) -> Option<&[u8]> {
        self.section_for_rva(rva)?.data_at_rva(rva, len)
    }
}

/// Parse a PE executable or original Xbox XBE without a PDB: sections + image
/// base + base relocations. XBE images have an empty base-relocation list.
pub fn load_pe_image(exe_data: &[u8]) -> Result<PeImage> {
    if exe_data.get(..4) == Some(b"XBEH") {
        return load_xbe_image(exe_data);
    }

    let (arch, image_base, sections) = parse_pe_sections(exe_data)?;
    let base_relocations = parse_base_relocations(&sections, image_base);
    Ok(PeImage {
        arch,
        image_base,
        sections,
        base_relocations,
    })
}

/// Parse an original Xbox executable into the image representation used by the
/// IDA importer.
///
/// XBE is not a PE file on disk: it has its own header and section table, but
/// the section contents are still mapped into a 32-bit x86 address space.  An
/// XBE has no PE base-relocation directory, so absolute relocations are
/// supplied by IDA's fixup export instead.
pub fn load_xbe_image(xbe_data: &[u8]) -> Result<PeImage> {
    const XBE_HEADER_SIZE: usize = 0x178;
    const XBE_SECTION_SIZE: usize = 0x38;
    const XBE_MAGIC: &[u8; 4] = b"XBEH";

    if xbe_data.len() < XBE_HEADER_SIZE {
        return Err(anyhow!("file too small for XBE header"));
    }
    if &xbe_data[..4] != XBE_MAGIC {
        return Err(anyhow!("not an XBE file (no XBEH signature)"));
    }

    // XBE image-header fields.  All addresses in the header are virtual
    // addresses; section raw addresses are file offsets.
    let image_base = read_u32(xbe_data, 0x104, "XBE base address")? as u64;
    let size_of_headers = read_u32(xbe_data, 0x108, "XBE header size")? as u64;
    let size_of_image = read_u32(xbe_data, 0x10c, "XBE image size")? as u64;
    let section_count = read_u32(xbe_data, 0x11c, "XBE section count")? as usize;
    let section_headers_addr = read_u32(xbe_data, 0x120, "XBE section headers address")? as u64;

    if image_base == 0 {
        return Err(anyhow!("XBE has a zero image base"));
    }
    let headers_end = image_base
        .checked_add(size_of_headers)
        .ok_or_else(|| anyhow!("XBE header address range overflows"))?;
    if section_headers_addr < image_base || section_headers_addr >= headers_end {
        return Err(anyhow!(
            "XBE section headers address {section_headers_addr:#x} is outside the header image"
        ));
    }
    let section_headers_offset = (section_headers_addr - image_base) as usize;
    let section_headers_len = section_count
        .checked_mul(XBE_SECTION_SIZE)
        .ok_or_else(|| anyhow!("XBE section table size overflows"))?;
    let section_headers_end = section_headers_offset
        .checked_add(section_headers_len)
        .ok_or_else(|| anyhow!("XBE section table offset overflows"))?;
    if section_headers_end > xbe_data.len() || section_headers_end as u64 > size_of_headers as u64 {
        return Err(anyhow!("XBE section headers are out of bounds"));
    }

    let mut sections = Vec::with_capacity(section_count);
    for index in 0..section_count {
        let offset = section_headers_offset + index * XBE_SECTION_SIZE;
        let header = &xbe_data[offset..offset + XBE_SECTION_SIZE];

        let flags = u32::from_le_bytes(header[0..4].try_into().unwrap());
        let virtual_address = u32::from_le_bytes(header[4..8].try_into().unwrap()) as u64;
        let virtual_size = u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;
        let raw_address = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        let raw_size = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        let section_name_addr = u32::from_le_bytes(header[20..24].try_into().unwrap()) as u64;

        let va = virtual_address;
        let section_end = va
            .checked_add(virtual_size)
            .ok_or_else(|| anyhow!("XBE section {index} virtual range overflows"))?;
        if size_of_image != 0 {
            let image_end = image_base
                .checked_add(size_of_image)
                .ok_or_else(|| anyhow!("XBE image address range overflows"))?;
            if va < image_base || section_end > image_end {
                return Err(anyhow!(
                    "XBE section {index} virtual range {va:#x}..{section_end:#x} is outside the image"
                ));
            }
        }

        let raw_end = raw_address
            .checked_add(raw_size)
            .ok_or_else(|| anyhow!("XBE section {index} raw range overflows"))?;
        if raw_end > xbe_data.len() {
            return Err(anyhow!("XBE section {index} raw data is out of bounds"));
        }

        let mut data = xbe_data[raw_address..raw_end].to_vec();
        let virtual_size_usize = usize::try_from(virtual_size)
            .map_err(|_| anyhow!("XBE section {index} virtual size is too large"))?;
        if data.len() < virtual_size_usize {
            data.resize(virtual_size_usize, 0);
        }

        let fallback_name = if flags & 0x04 != 0 {
            ".text"
        } else if flags & 0x01 != 0 {
            ".data"
        } else {
            ".rdata"
        };
        let mut name =
            xbe_cstr_at_virtual(xbe_data, image_base, size_of_headers, section_name_addr)
                .unwrap_or_else(|| fallback_name.to_string());
        if name.is_empty() {
            name = fallback_name.to_string();
        }
        if sections
            .iter()
            .any(|section: &PeSection| section.name == name)
        {
            name = format!("{name}_{index}");
        }

        // Keep PE-style characteristics for callers that inspect them, while
        // retaining the original XBE flags in the low bits.  The importer
        // primarily uses the virtual/raw address mapping and the IDA model's
        // section permissions.
        let mut characteristics = flags;
        if flags & 0x04 != 0 {
            characteristics |= 0x0000_0020 | 0x2000_0000;
        }
        if flags & 0x01 != 0 {
            characteristics |= 0x4000_0000 | 0x8000_0000;
        }
        if raw_size == 0 {
            characteristics |= 0x0000_0080;
        }

        sections.push(PeSection {
            name,
            rva: va.checked_sub(image_base).ok_or_else(|| {
                anyhow!("XBE section {index} virtual address precedes image base")
            })?,
            va,
            virtual_size,
            data,
            characteristics,
        });
    }

    Ok(PeImage {
        arch: PeArch::X86,
        image_base,
        sections,
        base_relocations: Vec::new(),
    })
}

fn read_u32(data: &[u8], offset: usize, field: &str) -> Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow!("{field} is out of bounds"))?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

/// Read an XBE virtual-addressed C string from the header image.  Section
/// names are normally stored in the header area; keeping this helper limited
/// to that area avoids accidentally interpreting an invalid name pointer as a
/// section's code or data.
fn xbe_cstr_at_virtual(
    data: &[u8],
    image_base: u64,
    size_of_headers: u64,
    address: u64,
) -> Option<String> {
    let offset = address
        .checked_sub(image_base)
        .and_then(|offset| usize::try_from(offset).ok())?;
    let header_size = usize::try_from(size_of_headers).ok()?;
    if offset >= header_size || offset >= data.len() {
        return None;
    }
    let end = data[offset..header_size.min(data.len())]
        .iter()
        .position(|&byte| byte == 0)
        .map(|relative| offset + relative)
        .unwrap_or_else(|| header_size.min(data.len()));
    Some(String::from_utf8_lossy(&data[offset..end]).into_owned())
}

// ---------------------------------------------------------------------------
// PE header parsing
// ---------------------------------------------------------------------------

fn parse_pe_sections(exe_data: &[u8]) -> Result<(PeArch, u64, Vec<PeSection>)> {
    if exe_data.len() < 0x40 {
        return Err(anyhow!("file too small for PE header"));
    }
    if &exe_data[0..2] != b"MZ" {
        return Err(anyhow!("not a PE file (no MZ signature)"));
    }

    let e_lfanew = u32::from_le_bytes(exe_data[0x3c..0x40].try_into().unwrap()) as usize;
    if e_lfanew + 4 > exe_data.len() {
        return Err(anyhow!("e_lfanew out of bounds"));
    }
    if &exe_data[e_lfanew..e_lfanew + 4] != b"PE\0\0" {
        return Err(anyhow!("PE signature not found"));
    }

    // COFF file header: 20 bytes starting at e_lfanew + 4.
    let coff_off = e_lfanew + 4;
    if coff_off + 20 > exe_data.len() {
        return Err(anyhow!("COFF header out of bounds"));
    }
    let machine = u16::from_le_bytes(exe_data[coff_off..coff_off + 2].try_into().unwrap());
    let arch = match machine {
        0x8664 => PeArch::X86_64,
        0x014C => PeArch::X86,
        other => {
            return Err(anyhow!(
                "unsupported machine type 0x{:04x} (only AMD64 and I386 supported)",
                other
            ))
        }
    };
    let num_sections =
        u16::from_le_bytes(exe_data[coff_off + 2..coff_off + 4].try_into().unwrap()) as usize;
    let opt_header_size =
        u16::from_le_bytes(exe_data[coff_off + 16..coff_off + 18].try_into().unwrap()) as usize;

    // Optional header starts right after the COFF file header.
    let opt_off = coff_off + 20;
    if opt_off + 2 > exe_data.len() {
        return Err(anyhow!("optional header offset out of bounds"));
    }
    let magic = u16::from_le_bytes(exe_data[opt_off..opt_off + 2].try_into().unwrap());

    // ImageBase offset and size differ between PE32 and PE32+.
    //   PE32  (0x010B): ImageBase at opt_off+28, 4 bytes (u32)
    //   PE32+ (0x020B): ImageBase at opt_off+24, 8 bytes (u64)
    let image_base: u64 = match magic {
        0x020B => {
            if opt_off + 32 > exe_data.len() {
                return Err(anyhow!("PE32+ optional header too short for ImageBase"));
            }
            u64::from_le_bytes(exe_data[opt_off + 24..opt_off + 32].try_into().unwrap())
        }
        0x010B => {
            if opt_off + 32 > exe_data.len() {
                return Err(anyhow!("PE32 optional header too short for ImageBase"));
            }
            u32::from_le_bytes(exe_data[opt_off + 28..opt_off + 32].try_into().unwrap()) as u64
        }
        other => {
            return Err(anyhow!(
                "unsupported PE optional-header magic 0x{:04x}",
                other
            ))
        }
    };

    // Section headers start after the optional header.
    let sections_off = opt_off + opt_header_size;
    let section_entry_size = 40usize;
    if sections_off + num_sections * section_entry_size > exe_data.len() {
        return Err(anyhow!("section headers out of bounds"));
    }

    let mut sections = Vec::with_capacity(num_sections);
    for i in 0..num_sections {
        let hdr = &exe_data[sections_off + i * section_entry_size..];
        let name_bytes = &hdr[0..8];
        let name = String::from_utf8_lossy(name_bytes)
            .trim_end_matches('\0')
            .to_string();

        let virtual_size = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as u64;
        let virtual_address = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as u64;
        let size_of_raw_data = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        let ptr_to_raw_data = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        let characteristics = u32::from_le_bytes(hdr[36..40].try_into().unwrap());

        let rva = virtual_address;
        let va = image_base + rva;

        let mut data = if ptr_to_raw_data == 0 || size_of_raw_data == 0 {
            Vec::new()
        } else {
            let end = ptr_to_raw_data + size_of_raw_data;
            if end > exe_data.len() {
                return Err(anyhow!("section '{}' raw data out of bounds", name));
            }
            exe_data[ptr_to_raw_data..end].to_vec()
        };
        if (data.len() as u64) < virtual_size {
            data.resize(virtual_size as usize, 0);
        }

        sections.push(PeSection {
            name,
            rva,
            va,
            virtual_size,
            data,
            characteristics,
        });
    }

    Ok((arch, image_base, sections))
}

// ---------------------------------------------------------------------------
// Base relocations (.reloc section)
// ---------------------------------------------------------------------------

fn parse_base_relocations(sections: &[PeSection], image_base: u64) -> Vec<BaseReloc> {
    let Some(reloc_section) = sections.iter().find(|s| s.name == ".reloc") else {
        return Vec::new();
    };
    let data = &reloc_section.data;
    let mut offset = 0usize;
    let mut relocs = Vec::new();

    while offset + 8 <= data.len() {
        let page_rva = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as u64;
        let block_size =
            u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;

        if block_size < 8 {
            break;
        }
        let entries_end = offset + block_size;
        if entries_end > data.len() {
            break;
        }

        let entries = &data[offset + 8..entries_end];
        for chunk in entries.chunks_exact(2) {
            let type_offset = u16::from_le_bytes(chunk.try_into().unwrap());
            let reloc_type = (type_offset >> 12) as u8;
            let page_offset = (type_offset & 0x0FFF) as u64;

            // Skip padding entries (type 0 = IMAGE_REL_BASED_ABSOLUTE).
            if reloc_type == 0 {
                continue;
            }

            let va = image_base + page_rva + page_offset;
            let kind = match reloc_type {
                3 => BaseRelocKind::HighLow,
                10 => BaseRelocKind::Dir64,
                other => BaseRelocKind::Other(other),
            };
            relocs.push(BaseReloc { va, kind });
        }

        offset = entries_end;
    }

    relocs
}

// ---------------------------------------------------------------------------
// Import table (IAT) parsing
// ---------------------------------------------------------------------------

fn parse_imports(
    exe_data: &[u8],
    sections: &[PeSection],
    image_base: u64,
    arch: PeArch,
) -> HashMap<u64, String> {
    match try_parse_imports(exe_data, sections, image_base, arch) {
        Ok(map) => map,
        Err(e) => {
            tracing::warn!("import table parse failed (ignoring): {e:#}");
            HashMap::new()
        }
    }
}

fn try_parse_imports(
    exe_data: &[u8],
    sections: &[PeSection],
    image_base: u64,
    arch: PeArch,
) -> Result<HashMap<u64, String>> {
    if exe_data.len() < 0x40 {
        return Ok(HashMap::new());
    }
    let e_lfanew = u32::from_le_bytes(exe_data[0x3c..0x40].try_into().unwrap()) as usize;
    let coff_off = e_lfanew + 4;
    let opt_off = coff_off + 20;

    // DataDirectory layout in the optional header:
    //   PE32  (32-bit): DataDirectory starts at opt_off + 96;  [1] at opt_off + 104
    //   PE32+ (64-bit): DataDirectory starts at opt_off + 112; [1] at opt_off + 120
    let import_dir_off = match arch {
        PeArch::X86_64 => opt_off + 112 + 8,
        PeArch::X86 => opt_off + 96 + 8,
    };
    if import_dir_off + 8 > exe_data.len() {
        return Ok(HashMap::new());
    }
    let import_rva = u32::from_le_bytes(
        exe_data[import_dir_off..import_dir_off + 4]
            .try_into()
            .unwrap(),
    );
    if import_rva == 0 {
        return Ok(HashMap::new());
    }

    // Thunk entry width: 8 bytes for PE32+, 4 bytes for PE32.
    let thunk_width: u64 = match arch {
        PeArch::X86_64 => 8,
        PeArch::X86 => 4,
    };

    let mut map = HashMap::new();
    let mut desc_rva = import_rva as u64;
    // IMAGE_IMPORT_DESCRIPTOR is 20 bytes.
    while let Some(desc) = rva_slice(sections, desc_rva, 20) {
        let original_first_thunk = u32::from_le_bytes(desc[0..4].try_into().unwrap()) as u64;
        let name_rva = u32::from_le_bytes(desc[12..16].try_into().unwrap()) as u64;
        let first_thunk = u32::from_le_bytes(desc[16..20].try_into().unwrap()) as u64;

        if original_first_thunk == 0 && first_thunk == 0 {
            break;
        }

        let hint_rva = if original_first_thunk != 0 {
            original_first_thunk
        } else {
            first_thunk
        };

        let mut entry_idx = 0u64;
        while let Some(thunk_bytes) = rva_slice(
            sections,
            hint_rva + entry_idx * thunk_width,
            thunk_width as usize,
        ) {
            // Read thunk as a native-width integer, zero-extended to u64.
            let thunk: u64 = match arch {
                PeArch::X86_64 => u64::from_le_bytes(thunk_bytes.try_into().unwrap()),
                PeArch::X86 => u32::from_le_bytes(thunk_bytes.try_into().unwrap()) as u64,
            };
            if thunk == 0 {
                break;
            }

            let iat_va = image_base + first_thunk + entry_idx * thunk_width;

            // High bit signals ordinal import; bit width depends on arch.
            let ordinal_bit = match arch {
                PeArch::X86_64 => 63,
                PeArch::X86 => 31,
            };
            let func_name = if (thunk >> ordinal_bit) & 1 == 1 {
                let ordinal = thunk as u16;
                let dll = rva_cstr(sections, name_rva).unwrap_or_default();
                format!("{}#{}", dll, ordinal)
            } else {
                // IMAGE_IMPORT_BY_NAME: 2-byte hint + name string.
                let ibn_rva = thunk & !(1u64 << ordinal_bit);
                rva_cstr(sections, ibn_rva + 2).unwrap_or_default()
            };

            if !func_name.is_empty() {
                map.insert(iat_va, format!("__imp_{}", func_name));
            }
            entry_idx += 1;
        }

        desc_rva += 20;
    }

    Ok(map)
}

// ---------------------------------------------------------------------------
// RVA helpers
// ---------------------------------------------------------------------------

pub(crate) fn rva_slice(sections: &[PeSection], rva: u64, len: usize) -> Option<&[u8]> {
    for s in sections {
        if rva >= s.rva && rva + len as u64 <= s.rva + s.virtual_size {
            let off = (rva - s.rva) as usize;
            return s.data.get(off..off + len);
        }
    }
    None
}

pub(crate) fn rva_cstr(sections: &[PeSection], rva: u64) -> Option<String> {
    for s in sections {
        if rva >= s.rva && rva < s.rva + s.virtual_size {
            let off = (rva - s.rva) as usize;
            let bytes = &s.data[off..];
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            return Some(String::from_utf8_lossy(&bytes[..end]).into_owned());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    #[test]
    fn loads_xbe_sections_as_an_x86_image() {
        let base = 0x10000u32;
        let headers_size = 0x400u32;
        let mut bytes = vec![0u8; 0x420];
        bytes[..4].copy_from_slice(b"XBEH");
        put_u32(&mut bytes, 0x104, base);
        put_u32(&mut bytes, 0x108, headers_size);
        put_u32(&mut bytes, 0x10c, 0x30000);
        put_u32(&mut bytes, 0x11c, 2);
        put_u32(&mut bytes, 0x120, base + 0x178);

        bytes[0x200..0x206].copy_from_slice(b".text\0");
        bytes[0x208..0x20e].copy_from_slice(b".data\0");

        // Section 0: executable code with a zero-filled virtual tail.
        put_u32(&mut bytes, 0x178, 0x06);
        put_u32(&mut bytes, 0x17c, base + 0x1000);
        put_u32(&mut bytes, 0x180, 8);
        put_u32(&mut bytes, 0x184, 0x400);
        put_u32(&mut bytes, 0x188, 4);
        put_u32(&mut bytes, 0x18c, base + 0x200);
        bytes[0x400..0x404].copy_from_slice(&[0x55, 0x8b, 0xec, 0xc3]);

        // Section 1: writable data.
        put_u32(&mut bytes, 0x1b0, 0x01);
        put_u32(&mut bytes, 0x1b4, base + 0x2000);
        put_u32(&mut bytes, 0x1b8, 4);
        put_u32(&mut bytes, 0x1bc, 0x410);
        put_u32(&mut bytes, 0x1c0, 4);
        put_u32(&mut bytes, 0x1c4, base + 0x208);
        bytes[0x410..0x414].copy_from_slice(&[1, 2, 3, 4]);

        let image = load_pe_image(&bytes).unwrap();
        assert_eq!(image.arch, PeArch::X86);
        assert_eq!(image.image_base, base as u64);
        assert!(image.base_relocations.is_empty());
        assert_eq!(image.sections[0].name, ".text");
        assert_eq!(image.sections[0].rva, 0x1000);
        assert_eq!(
            image.sections[0].data,
            vec![0x55, 0x8b, 0xec, 0xc3, 0, 0, 0, 0]
        );
        assert_eq!(image.data_at_rva(0x1000, 8).unwrap().len(), 8);
        assert_eq!(image.sections[1].name, ".data");
        assert_eq!(image.data_at_rva(0x2000, 4).unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn rejects_xbe_section_raw_data_outside_file() {
        let base = 0x10000u32;
        let mut bytes = vec![0u8; 0x200];
        bytes[..4].copy_from_slice(b"XBEH");
        put_u32(&mut bytes, 0x104, base);
        put_u32(&mut bytes, 0x108, 0x200);
        put_u32(&mut bytes, 0x10c, 0x20000);
        put_u32(&mut bytes, 0x11c, 1);
        put_u32(&mut bytes, 0x120, base + 0x178);
        put_u32(&mut bytes, 0x178, 0x04);
        put_u32(&mut bytes, 0x17c, base + 0x1000);
        put_u32(&mut bytes, 0x180, 4);
        put_u32(&mut bytes, 0x184, 0x1ff);
        put_u32(&mut bytes, 0x188, 4);

        let error = match load_xbe_image(&bytes) {
            Ok(_) => panic!("malformed XBE unexpectedly loaded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("raw data is out of bounds"));
    }
}
