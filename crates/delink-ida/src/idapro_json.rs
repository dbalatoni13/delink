//! User-editable object grouping for the IDA workflow.
//!
//! Each output object lists its function start addresses and may also own
//! half-open virtual-address ranges of whole functions, `.rdata`, `.data`,
//! logical `.bss`, and custom output sections.
//!
//! ```json
//! {
//!   "Alchemy.dll.obj": {
//!     "functions": ["0x1000", "0x1030"],
//!     "function_ranges": [["0x2000", "0x23E8"]],
//!     "rdata": [["0x11000", "0x11080"]],
//!     "data": [["0x21000", "0x21040"]],
//!     "bss": [["0x22000", "0x22400"]],
//!     "custom_sections": {
//!       ".text$init": { "ranges": [["0x3000", "0x3100"]] },
//!       ".globals": { "kind": "data", "ranges": [["0x21040", "0x21080"], ["0x21100", "0x21120"]] }
//!     }
//!   }
//! }
//! ```
//!
//! Custom section names starting with `.text` default to `code`.
//! The old `{ "file.obj": [address, ...] }` form remains accepted.

use std::collections::BTreeMap;
use std::ops::Range;

use serde::de::Error as _;
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};

use crate::IdaModel;

/// A half-open IDA virtual-address range, serialized as the JSON pair
/// `[start, end]` representing `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataRange(pub [u64; 2]);

impl Serialize for DataRange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(2))?;
        seq.serialize_element(&format!("0x{:X}", self.0[0]))?;
        seq.serialize_element(&format!("0x{:X}", self.0[1]))?;
        seq.end()
    }
}

impl<'de> Deserialize<'de> for DataRange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let values = Vec::<HexU64>::deserialize(deserializer)?;
        let [start, end] = values.as_slice() else {
            return Err(D::Error::custom("address range must contain two values"));
        };
        Ok(Self([start.0, end.0]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HexU64(u64);

impl<'de> Deserialize<'de> for HexU64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(text) => {
                let hex = text
                    .strip_prefix("0x")
                    .or_else(|| text.strip_prefix("0X"))
                    .ok_or_else(|| D::Error::custom("address must be hexadecimal"))?;
                u64::from_str_radix(hex, 16)
                    .map(Self)
                    .map_err(D::Error::custom)
            }
            serde_json::Value::Number(number) => number
                .as_u64()
                .map(Self)
                .ok_or_else(|| D::Error::custom("address must be nonnegative")),
            _ => Err(D::Error::custom(
                "address must be hexadecimal or an integer",
            )),
        }
    }
}

fn deserialize_u64_vec<'de, D>(deserializer: D) -> Result<Vec<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Vec::<HexU64>::deserialize(deserializer)?
        .into_iter()
        .map(|value| value.0)
        .collect())
}

fn serialize_u64_vec<S>(values: &[u64], serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut seq = serializer.serialize_seq(Some(values.len()))?;
    for value in values {
        seq.serialize_element(&format!("0x{value:X}"))?;
    }
    seq.end()
}

impl DataRange {
    pub fn range(self) -> Range<u64> {
        self.0[0]..self.0[1]
    }
}

/// The output section type and the rules used to emit its ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CustomSectionKind {
    Code,
    Rdata,
    Data,
    Bss,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomSection {
    pub kind: CustomSectionKind,
    pub ranges: Vec<DataRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
struct CustomSectionInput {
    #[serde(default, deserialize_with = "deserialize_explicit_kind")]
    kind: Option<CustomSectionKind>,
    ranges: Vec<DataRange>,
}

fn deserialize_explicit_kind<'de, D>(deserializer: D) -> Result<Option<CustomSectionKind>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    CustomSectionKind::deserialize(deserializer).map(Some)
}

/// Contents assigned to one output object.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ObjectGroup {
    #[serde(default, serialize_with = "serialize_u64_vec")]
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
    /// Custom output section names and the ranges assigned to each one.
    #[serde(default)]
    pub custom_sections: BTreeMap<String, CustomSection>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
struct DetailedObjectGroup {
    #[serde(default, deserialize_with = "deserialize_u64_vec")]
    functions: Vec<u64>,
    #[serde(default)]
    function_ranges: Vec<DataRange>,
    #[serde(default)]
    rdata: Vec<DataRange>,
    #[serde(default)]
    data: Vec<DataRange>,
    #[serde(default)]
    bss: Vec<DataRange>,
    #[serde(default)]
    custom_sections: BTreeMap<String, CustomSectionInput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
enum ObjectGroupInput {
    Detailed(DetailedObjectGroup),
    Functions(Vec<HexU64>),
}

impl<'de> Deserialize<'de> for ObjectGroup {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(match ObjectGroupInput::deserialize(deserializer)? {
            ObjectGroupInput::Detailed(group) => {
                let mut custom_sections = BTreeMap::new();
                for (name, section) in group.custom_sections {
                    let kind = match section.kind {
                        Some(kind) => kind,
                        None if name.starts_with(".text") => CustomSectionKind::Code,
                        None => {
                            return Err(D::Error::custom(format!(
                                "custom section {name:?} requires a kind"
                            )))
                        }
                    };
                    custom_sections.insert(
                        name,
                        CustomSection {
                            kind,
                            ranges: section.ranges,
                        },
                    );
                }
                Self {
                    functions: group.functions,
                    function_ranges: group.function_ranges,
                    rdata: group.rdata,
                    data: group.data,
                    bss: group.bss,
                    custom_sections,
                }
            }
            ObjectGroupInput::Functions(functions) => Self {
                functions: functions.into_iter().map(|address| address.0).collect(),
                ..Self::default()
            },
        })
    }
}

/// Output filename to its functions and section ranges.
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
        assert!(json["one.obj"].custom_sections.is_empty());
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
    fn accepts_hexadecimal_addresses() {
        let json: IdaproJson = serde_json::from_str(
            r#"{"one.obj":{"functions":["0x1000"],"rdata":[["0x2000","0x2010"]]}}"#,
        )
        .unwrap();
        assert_eq!(json["one.obj"].functions, vec![0x1000]);
        assert_eq!(json["one.obj"].rdata[0].range(), 0x2000..0x2010);
    }

    #[test]
    fn custom_sections_without_a_text_prefix_require_a_kind() {
        let input = r#"{"one.obj":{"custom_sections":{".extra":{"ranges":[[4096,4100]]}}}}"#;
        assert!(serde_json::from_str::<IdaproJson>(input).is_err());
    }

    #[test]
    fn explicit_kind_overrides_the_text_default() {
        let input = r#"{"one.obj":{"custom_sections":{".text.extra":{"kind":"data","ranges":[[4096,4100]]}}}}"#;
        let json: IdaproJson = serde_json::from_str(input).unwrap();
        assert_eq!(
            json["one.obj"].custom_sections[".text.extra"].kind,
            CustomSectionKind::Data
        );
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
        assert_eq!(value["one.obj"]["functions"], serde_json::json!(["0x1000"]));
        assert_eq!(value["one.obj"]["function_ranges"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["rdata"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["data"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["bss"], serde_json::json!([]));
        assert_eq!(value["one.obj"]["custom_sections"], serde_json::json!({}));
    }

    #[test]
    fn serializes_ranges_as_hexadecimal_strings() {
        let range = DataRange([0x401000, 0x401010]);
        assert_eq!(
            serde_json::to_value(range).unwrap(),
            serde_json::json!(["0x401000", "0x401010"])
        );
    }
}
