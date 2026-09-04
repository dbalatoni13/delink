//! User-editable function grouping: maps each output object filename to a list
//! of function start addresses.
//!
//! ```json
//! {
//!   "Alchemy.dll.obj": [
//!     268441600,
//!     268441648
//!   ]
//! }
//! ```
//!
//! `idapro.json` deliberately contains grouping information only.  Function
//! names, bounds/sizes, and visibility are always read from the authoritative
//! `delink.json` model.

use std::collections::BTreeMap;

use crate::IdaModel;

/// Output-filename → function start addresses.
pub type IdaproJson = BTreeMap<String, Vec<u64>>;

/// Build a default grouping: one function per output file (extension `ext`,
/// e.g. `"obj"` for COFF or `"o"` for ELF).
pub fn generate(model: &IdaModel, ext: &str) -> IdaproJson {
    let mut json: IdaproJson = BTreeMap::new();
    for f in &model.functions {
        if f.size() == 0 {
            continue;
        }
        let file = format!("{}.{ext}", sanitize_filename(&f.name));
        json.entry(file).or_default().push(f.start);
    }
    json
}

fn sanitize_filename(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, '_' | '-' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = s.trim_start_matches(['.', '_']);
    let truncated = &trimmed[..trimmed.len().min(200)];
    if truncated.is_empty() {
        "unknown".to_string()
    } else {
        truncated.to_string()
    }
}
