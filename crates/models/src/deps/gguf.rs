//! A GGUF file as a [`WeightSource`], and the writer that turns any source
//! into one. GGUF is what the local ecosystem ships: one file, quantised,
//! with the tokenizer and chat template inside.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file::{self, Content, Value};
use candle_core::quantized::QTensor;
use candle_core::Device;

use infy_kernel::{Error, Result};

use crate::deps::WeightSource;
use crate::logic::{MetaValue, Metadata};

/// A GGUF on disk. The header is parsed once; tensors are read on demand.
pub struct GgufFile {
    path: PathBuf,
    content: Content,
    meta: Metadata,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                Error::not_found(format!("model file {}", path.display()))
            } else {
                Error::wrap(format!("opening {}", path.display()), e)
            }
        })?;
        let mut reader = BufReader::new(file);
        let content = Content::read(&mut reader)
            .map_err(|e| Error::wrap(format!("reading GGUF header of {}", path.display()), e))?;
        let meta = content
            .metadata
            .iter()
            .map(|(k, v)| (k.clone(), from_candle(v)))
            .collect();
        Ok(Self {
            path,
            content,
            meta,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total bytes of tensor data, for a catalogue listing.
    pub fn tensor_bytes(&self) -> usize {
        self.content
            .tensor_infos
            .values()
            .map(|i| i.shape.elem_count() / i.ggml_dtype.block_size() * i.ggml_dtype.type_size())
            .sum()
    }
}

impl WeightSource for GgufFile {
    fn metadata(&self) -> &Metadata {
        &self.meta
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.content.tensor_infos.keys().cloned().collect();
        names.sort();
        names
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    fn tensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        let file = File::open(&self.path)
            .map_err(|e| Error::wrap(format!("reopening {}", self.path.display()), e))?;
        let mut reader = BufReader::new(file);
        self.content.tensor(&mut reader, name, device).map_err(|e| {
            Error::wrap(
                format!("reading tensor {name} from {}", self.path.display()),
                e,
            )
        })
    }
}

/// Write any source as a GGUF. Tensors are read on the CPU and stored as
/// they come. Metadata arrays that are empty are skipped: the format has no
/// way to say "an array of nothing in particular".
pub fn write_gguf(path: impl AsRef<Path>, source: &dyn WeightSource) -> Result<()> {
    let path = path.as_ref();
    let mut meta: Vec<(String, Value)> = source
        .metadata()
        .iter()
        .filter(|(_, v)| !matches!(v, MetaValue::Array(a) if a.is_empty()))
        .map(|(k, v)| (k.clone(), to_candle(v)))
        .collect();
    meta.sort_by(|a, b| a.0.cmp(&b.0));
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    for name in source.tensor_names() {
        let t = source.tensor(&name, &Device::Cpu)?;
        tensors.push((name, t));
    }
    let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut file =
        File::create(path).map_err(|e| Error::wrap(format!("creating {}", path.display()), e))?;
    gguf_file::write(&mut file, &meta_refs, &tensor_refs)
        .map_err(|e| Error::wrap(format!("writing {}", path.display()), e))
}

fn from_candle(v: &Value) -> MetaValue {
    match v {
        Value::U8(x) => MetaValue::U8(*x),
        Value::I8(x) => MetaValue::I8(*x),
        Value::U16(x) => MetaValue::U16(*x),
        Value::I16(x) => MetaValue::I16(*x),
        Value::U32(x) => MetaValue::U32(*x),
        Value::I32(x) => MetaValue::I32(*x),
        Value::U64(x) => MetaValue::U64(*x),
        Value::I64(x) => MetaValue::I64(*x),
        Value::F32(x) => MetaValue::F32(*x),
        Value::F64(x) => MetaValue::F64(*x),
        Value::Bool(x) => MetaValue::Bool(*x),
        Value::String(s) => MetaValue::Str(s.clone()),
        Value::Array(a) => MetaValue::Array(a.iter().map(from_candle).collect()),
    }
}

fn to_candle(v: &MetaValue) -> Value {
    match v {
        MetaValue::U8(x) => Value::U8(*x),
        MetaValue::I8(x) => Value::I8(*x),
        MetaValue::U16(x) => Value::U16(*x),
        MetaValue::I16(x) => Value::I16(*x),
        MetaValue::U32(x) => Value::U32(*x),
        MetaValue::I32(x) => Value::I32(*x),
        MetaValue::U64(x) => Value::U64(*x),
        MetaValue::I64(x) => Value::I64(*x),
        MetaValue::F32(x) => Value::F32(*x),
        MetaValue::F64(x) => Value::F64(*x),
        MetaValue::Bool(x) => Value::Bool(*x),
        MetaValue::Str(s) => Value::String(s.clone()),
        MetaValue::Array(a) => Value::Array(a.iter().map(to_candle).collect()),
    }
}
