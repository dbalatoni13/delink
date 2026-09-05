//! User-editable object grouping for the IDA workflow.
//!
//! Each output object lists its function start addresses and may also own
//! half-open virtual-address ranges of whole functions, `.rdata`, `.data`, and
//! logical `.bss`:
//!
//! ```json
//! {
//!   "Alchemy.dll.obj": {
//!     "functions": [268441600, 268441648],
//!     "function_ranges": [[268442000, 268443000]],
//!     "rdata": [[268500992, 268501120]],
//!     "data": [[268566528, 268566592]],
//!     "bss": [[268570624, 268571648]]
//!   }
//! }
//! ```
//!
//! The old `{ "file.obj": [address, ...] }` form remains accepted.

use std::collections::BTreeMap;
use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::IdaModel;

/// A half-open IDA virtual-address range, serialized as the JSON pair
/// `[start, end]` representing `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DataRange(pub [u64; 2]);

impl DataRange {
    pub fn range(self) -> Range<u64> {
        self.0[0]..self.0[1]
    }
}

/// Contents assigned to one output object.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ObjectGroup {
    #[serde(default)]
    pub functions: Vec<u64>,
    /// Half-open ranges selecting whole functions by their `[start, end)`
    /// bounds. A range may not cut through a function.
    #[serde(default)]
    pub function_ranges: Vec<DataRange>,
    #[serde(default)]
    pub rdata: Vec<DataRange>,
    #[serde(default)]
    pub data: Vec<DataRange>,
    /// Half-open ranges emitted as zero-filled `.bss`. These may select an
    /// IDA BSS segment or a zero-initialized tail that IDA reports as DATA.
    #[serde(default)]
    pub bss: Vec<DataRange>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
struct DetailedObjectGroup {
    #[serde(default)]
    functions: Vec<u64>,
    #[serde(default)]
    function_ranges: Vec<DataRange>,
    #[serde(default)]
    rdata: Vec<DataRange>,
    #[serde(default)]
    data: Vec<DataRange>,
    #[serde(default)]
    bss: Vec<DataRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum ObjectGroupInput {
    Detailed(DetailedObjectGroup),
    Functions(Vec<u64>),
}

impl<'de> Deserialize<'de> for ObjectGroup {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match ObjectGroupInput::deserialize(deserializer)? {
            ObjectGroupInput::Detailed(group) => Self {
                functions: group.functions,
                function_ranges: group.function_ranges,
                rdata: group.rdata,
                data: group.data,
                bss: group.bss,
            },
            ObjectGroupInput::Functions(functions) => Self {
                functions,
                ..Self::default()
            },
        })
    }
}

/// Output filename to its explicit/ranged functions and initialized-data
/// ranges.
pub type IdaproJson = BTreeMap<String, ObjectGroup>;

/// Build a default grouping: one function per output file (extension `ext`,
/// e.g. `"obj"` for COFF or `"o"` for ELF).
pub fn generate(model: &IdaModel, ext: &str) -> IdaproJson {
    let mut json = IdaproJson::new();
    for f in &model.functions {
        if f.size() == 0 {
            continue;
        }
        let file = format!("{}.{ext}", sanitize_filename(&f.name));
        json.entry(file).or_default().functions.push(f.start);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_legacy_function_arrays() {
        let json: IdaproJson = serde_json::from_str(r#"{"one.obj":[4096,8192]}"#).unwrap();
        assert_eq!(json["one.obj"].functions, vec![4096, 8192]);
        assert!(json["one.obj"].rdata.is_empty());
        assert!(json["one.obj"].bss.is_empty());
        assert!(json["one.obj"].function_ranges.is_empty());
    }

    #[test]
    fn accepts_data_ranges() {
        let json: IdaproJson = serde_json::from_str(
            r#"{"one.obj":{"functions":[4096],"function_ranges":[[4097,8192]],"rdata":[[8192,8208]],"data":[[12288,12320]],"bss":[[16384,16400]]}}"#,
        )
        .unwrap();
        assert_eq!(json["one.obj"].rdata[0].range(), 8192..8208);
        assert_eq!(json["one.obj"].data[0].range(), 12288..12320);
        assert_eq!(json["one.obj"].function_ranges[0].range(), 4097..8192);
        assert_eq!(json["one.obj"].bss[0].range(), 16384..16400);
    }

    #[test]
    fn serializes_the_editable_data_fields() {
        let json = IdaproJson::from([(
            "one.obj".to_string(),
            ObjectGroup {
                functions: vec![4096],
                ..ObjectGroup::default()
            },
        )]);
        let value = serde_json::to_value(json).unwrap();
        assert_eq!(value["one.obj"]["functions"], serde_json::json!([4096]));
        assert_eq!(value["one.obj"]["function_ranges"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["rdata"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["data"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["bss"], serde_json::json!([]));
    }
}
