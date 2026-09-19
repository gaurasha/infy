//! Pure logic: metadata → configuration and vocabulary, RoPE tables, the
//! causal mask, the list of tensors an architecture needs. No tensors, no
//! device, no I/O, so every function here is tested by a table.

use std::collections::HashMap;
use std::ops::Range;

use infy_kernel::{Error, Result, TokenId, Vocabulary};

/// One metadata value, as GGUF stores them. Mirrors the file's own type set
/// rather than candle's so that this module, and the adapters that fill it,
/// stay independent of the tensor library's representation.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    /// Any integer, widened. GGUF writers are inconsistent about whether a
    /// count is `u32` or `u64`; readers should not care.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(v.into()),
            MetaValue::U16(v) => Some(v.into()),
            MetaValue::U32(v) => Some(v.into()),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) => u64::try_from(v).ok(),
            MetaValue::I16(v) => u64::try_from(v).ok(),
            MetaValue::I32(v) => u64::try_from(v).ok(),
            MetaValue::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            MetaValue::F32(v) => Some(v.into()),
            MetaValue::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            MetaValue::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetaValue]> {
        match self {
            MetaValue::Array(v) => Some(v),
            _ => None,
        }
    }
}

/// The flat key → value map a checkpoint carries.
pub type Metadata = HashMap<String, MetaValue>;

/// The architectures this crate can run. The name is the GGUF
/// `general.architecture` value, which is what a file actually says rather
/// than what its README calls it: Mistral, SmolLM and TinyLlama files all say
/// `llama`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,
    Qwen2,
}

impl Architecture {
    pub const SUPPORTED: &'static [&'static str] = &["llama", "qwen2"];

    pub fn parse(name: &str) -> Result<Self> {
        match name {
            "llama" => Ok(Architecture::Llama),
            "qwen2" => Ok(Architecture::Qwen2),
            other => Err(Error::invalid(format!(
                "architecture {other:?} is not supported yet; infy runs {}",
                Self::SUPPORTED.join(", ")
            ))),
        }
    }

    pub fn gguf_name(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen2 => "qwen2",
        }
    }

    /// Whether the file's Q and K weights expect interleaved-pair RoPE.
    ///
    /// llama.cpp's converter permutes Llama's Q and K projections so that its
    /// "normal" rotary embedding (rotating adjacent pairs) is correct; Qwen2
    /// is left unpermuted and uses the NeoX convention (rotating the two
    /// halves). Applying the wrong one produces confident nonsense, so this
    /// is a property of the architecture, not an option.
    pub fn rope_interleaved(self) -> bool {
        match self {
            Architecture::Llama => true,
            Architecture::Qwen2 => false,
        }
    }
}

/// How positions are stretched before RoPE.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    None,
    /// Positions divided by `factor` (the "linear" scheme).
    Linear {
        factor: f32,
    },
}

/// Everything the architecture needs to know, read out of the metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub architecture: Architecture,
    pub name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
}

fn key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}

fn required_u64(meta: &Metadata, k: &str) -> Result<u64> {
    meta.get(k)
        .and_then(MetaValue::as_u64)
        .ok_or_else(|| Error::invalid(format!("metadata key {k:?} is missing or not an integer")))
}

fn optional_u64(meta: &Metadata, k: &str) -> Option<u64> {
    meta.get(k).and_then(MetaValue::as_u64)
}

fn optional_f32(meta: &Metadata, k: &str) -> Option<f32> {
    meta.get(k).and_then(MetaValue::as_f64).map(|v| v as f32)
}

impl ModelConfig {
    /// Read the configuration a llama.cpp-converted GGUF carries.
    pub fn from_metadata(meta: &Metadata) -> Result<Self> {
        let arch_name = meta
            .get("general.architecture")
            .and_then(MetaValue::as_str)
            .ok_or_else(|| Error::invalid("metadata has no general.architecture"))?;
        let architecture = Architecture::parse(arch_name)?;
        let a = architecture.gguf_name();

        let hidden_size = required_u64(meta, &key(a, "embedding_length"))? as usize;
        let num_layers = required_u64(meta, &key(a, "block_count"))? as usize;
        let num_heads = required_u64(meta, &key(a, "attention.head_count"))? as usize;
        let num_kv_heads = optional_u64(meta, &key(a, "attention.head_count_kv"))
            .unwrap_or(num_heads as u64) as usize;
        let intermediate_size = required_u64(meta, &key(a, "feed_forward_length"))? as usize;
        let context_length = required_u64(meta, &key(a, "context_length"))? as usize;
        let rms_norm_eps =
            optional_f32(meta, &key(a, "attention.layer_norm_rms_epsilon")).unwrap_or(1e-5);
        let rope_theta = optional_f32(meta, &key(a, "rope.freq_base")).unwrap_or(10_000.0);
        let head_dim = optional_u64(meta, &key(a, "attention.key_length"))
            .or_else(|| optional_u64(meta, &key(a, "rope.dimension_count")))
            .map(|v| v as usize)
            .unwrap_or_else(|| {
                if num_heads == 0 {
                    0
                } else {
                    hidden_size / num_heads
                }
            });
        let vocab_size = optional_u64(meta, &key(a, "vocab_size"))
            .map(|v| v as usize)
            .or_else(|| meta.get("tokenizer.ggml.tokens").and_then(MetaValue::as_array).map(|t| t.len()))
            .ok_or_else(|| {
                Error::invalid(format!(
                    "cannot determine vocabulary size: neither {} nor tokenizer.ggml.tokens present",
                    key(a, "vocab_size")
                ))
            })?;
        let rope_scaling = match meta
            .get(&key(a, "rope.scaling.type"))
            .and_then(MetaValue::as_str)
        {
            None | Some("none") => RopeScaling::None,
            Some("linear") => RopeScaling::Linear {
                factor: optional_f32(meta, &key(a, "rope.scaling.factor")).unwrap_or(1.0),
            },
            Some(other) => {
                return Err(Error::invalid(format!(
                    "rope scaling type {other:?} is not supported yet (linear is)"
                )))
            }
        };
        let name = meta
            .get("general.name")
            .and_then(MetaValue::as_str)
            .unwrap_or(arch_name)
            .to_string();

        let cfg = Self {
            architecture,
            name,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            context_length,
            rms_norm_eps,
            rope_theta,
            rope_scaling,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The inverse of [`from_metadata`](Self::from_metadata), so an in-memory
    /// model can be written as a GGUF that reads back identically.
    pub fn to_metadata(&self) -> Metadata {
        let a = self.architecture.gguf_name();
        let mut m = Metadata::new();
        m.insert("general.architecture".into(), MetaValue::Str(a.into()));
        m.insert("general.name".into(), MetaValue::Str(self.name.clone()));
        let u = |v: usize| MetaValue::U32(v as u32);
        m.insert(key(a, "embedding_length"), u(self.hidden_size));
        m.insert(key(a, "block_count"), u(self.num_layers));
        m.insert(key(a, "attention.head_count"), u(self.num_heads));
        m.insert(key(a, "attention.head_count_kv"), u(self.num_kv_heads));
        m.insert(key(a, "feed_forward_length"), u(self.intermediate_size));
        m.insert(key(a, "context_length"), u(self.context_length));
        m.insert(key(a, "vocab_size"), u(self.vocab_size));
        m.insert(key(a, "rope.dimension_count"), u(self.head_dim));
        m.insert(
            key(a, "attention.layer_norm_rms_epsilon"),
            MetaValue::F32(self.rms_norm_eps),
        );
        m.insert(key(a, "rope.freq_base"), MetaValue::F32(self.rope_theta));
        if let RopeScaling::Linear { factor } = self.rope_scaling {
            m.insert(key(a, "rope.scaling.type"), MetaValue::Str("linear".into()));
            m.insert(key(a, "rope.scaling.factor"), MetaValue::F32(factor));
        }
        m
    }

    pub fn validate(&self) -> Result<()> {
        let bad = |what: &str| Err(Error::invalid(format!("model config: {what}")));
        if self.num_layers == 0 || self.num_heads == 0 || self.num_kv_heads == 0 {
            return bad("layer and head counts must be positive");
        }
        if !self.num_heads.is_multiple_of(self.num_kv_heads) {
            return bad(&format!(
                "head_count {} is not a multiple of head_count_kv {}",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return bad(&format!(
                "head_dim {} must be positive and even for RoPE",
                self.head_dim
            ));
        }
        if self.hidden_size == 0 || self.intermediate_size == 0 || self.vocab_size == 0 {
            return bad("embedding, feed-forward and vocabulary sizes must be positive");
        }
        if self.context_length == 0 {
            return bad("context_length must be positive");
        }
        Ok(())
    }

    pub fn rope_interleaved(&self) -> bool {
        self.architecture.rope_interleaved()
    }

    /// Bytes the KV cache holds for `tokens` positions at `elem_bytes` per
    /// element: two tensors per layer of `num_kv_heads × tokens × head_dim`.
    pub fn kv_cache_bytes(&self, tokens: usize, elem_bytes: usize) -> usize {
        2 * self.num_layers * self.num_kv_heads * tokens * self.head_dim * elem_bytes
    }

    /// Every tensor the architecture reads, with the shape it expects, in
    /// candle's `(out, in)` orientation. Optional entries may be absent from
    /// a file: `output.weight` (tied to the embedding when missing), the
    /// attention biases (Qwen2 has them, Llama does not) and
    /// `rope_freqs.weight` (present only for Llama 3.1+ rope scaling).
    pub fn tensor_spec(&self) -> Vec<TensorSpec> {
        let h = self.hidden_size;
        let kv = self.num_kv_heads * self.head_dim;
        let q = self.num_heads * self.head_dim;
        let f = self.intermediate_size;
        let mut out = vec![
            TensorSpec::required("token_embd.weight", vec![self.vocab_size, h]),
            TensorSpec::required("output_norm.weight", vec![h]),
            TensorSpec::optional("output.weight", vec![self.vocab_size, h]),
            TensorSpec::optional("rope_freqs.weight", vec![self.head_dim / 2]),
        ];
        for i in 0..self.num_layers {
            let p = format!("blk.{i}");
            out.push(TensorSpec::required(
                format!("{p}.attn_norm.weight"),
                vec![h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.attn_q.weight"),
                vec![q, h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.attn_k.weight"),
                vec![kv, h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.attn_v.weight"),
                vec![kv, h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.attn_output.weight"),
                vec![h, q],
            ));
            out.push(TensorSpec::optional(format!("{p}.attn_q.bias"), vec![q]));
            out.push(TensorSpec::optional(format!("{p}.attn_k.bias"), vec![kv]));
            out.push(TensorSpec::optional(format!("{p}.attn_v.bias"), vec![kv]));
            out.push(TensorSpec::required(
                format!("{p}.ffn_norm.weight"),
                vec![h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.ffn_gate.weight"),
                vec![f, h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.ffn_up.weight"),
                vec![f, h],
            ));
            out.push(TensorSpec::required(
                format!("{p}.ffn_down.weight"),
                vec![h, f],
            ));
        }
        out
    }

    /// The required tensors missing from `present`, for an error that names
    /// them instead of failing on the first one.
    pub fn missing_tensors(&self, present: &[String]) -> Vec<String> {
        self.tensor_spec()
            .into_iter()
            .filter(|s| s.required && !present.contains(&s.name))
            .map(|s| s.name)
            .collect()
    }
}

/// A tensor the architecture reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub required: bool,
}

impl TensorSpec {
    fn required(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            required: true,
        }
    }
    fn optional(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            required: false,
        }
    }
}

// --- vocabulary --------------------------------------------------------------

fn str_array(meta: &Metadata, k: &str) -> Vec<String> {
    meta.get(k)
        .and_then(MetaValue::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn token_id(meta: &Metadata, k: &str) -> Option<TokenId> {
    optional_u64(meta, k).and_then(|v| TokenId::try_from(v).ok())
}

/// Lift the `tokenizer.*` metadata into a [`Vocabulary`].
///
/// Every end-of-generation id the file names is collected: `eos_token_id`
/// plus, where present, `eot_token_id` and `eom_token_id` (Llama 3 ends a
/// turn with `<|eot_id|>`, not with its `eos`). Missing tokens are an error
/// because a model without a vocabulary cannot be talked to; everything else
/// is optional.
pub fn vocabulary_from_metadata(meta: &Metadata) -> Result<Vocabulary> {
    let tokens = str_array(meta, "tokenizer.ggml.tokens");
    if tokens.is_empty() {
        return Err(Error::invalid(
            "metadata has no tokenizer.ggml.tokens; this file carries no vocabulary",
        ));
    }
    let scores = meta
        .get("tokenizer.ggml.scores")
        .and_then(MetaValue::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default();
    let token_types = meta
        .get("tokenizer.ggml.token_type")
        .and_then(MetaValue::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_f64().map(|f| f as i32))
                .collect()
        })
        .unwrap_or_default();
    let mut eos_token_ids = Vec::new();
    for k in [
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.eot_token_id",
        "tokenizer.ggml.eom_token_id",
    ] {
        if let Some(id) = token_id(meta, k) {
            if !eos_token_ids.contains(&id) {
                eos_token_ids.push(id);
            }
        }
    }
    Ok(Vocabulary {
        model: meta
            .get("tokenizer.ggml.model")
            .and_then(MetaValue::as_str)
            .unwrap_or("")
            .to_string(),
        pre: meta
            .get("tokenizer.ggml.pre")
            .and_then(MetaValue::as_str)
            .map(str::to_string),
        tokens,
        scores,
        token_types,
        merges: str_array(meta, "tokenizer.ggml.merges"),
        bos_token_id: token_id(meta, "tokenizer.ggml.bos_token_id"),
        eos_token_ids,
        unk_token_id: token_id(meta, "tokenizer.ggml.unknown_token_id"),
        pad_token_id: token_id(meta, "tokenizer.ggml.padding_token_id"),
        add_bos_token: meta
            .get("tokenizer.ggml.add_bos_token")
            .and_then(MetaValue::as_bool),
        add_eos_token: meta
            .get("tokenizer.ggml.add_eos_token")
            .and_then(MetaValue::as_bool),
        chat_template: meta
            .get("tokenizer.chat_template")
            .and_then(MetaValue::as_str)
            .map(str::to_string),
    })
}

/// The inverse of [`vocabulary_from_metadata`]. Empty lists are omitted
/// because GGUF cannot represent an empty typed array.
pub fn vocabulary_to_metadata(v: &Vocabulary) -> Metadata {
    let mut m = Metadata::new();
    m.insert(
        "tokenizer.ggml.model".into(),
        MetaValue::Str(v.model.clone()),
    );
    if let Some(pre) = &v.pre {
        m.insert("tokenizer.ggml.pre".into(), MetaValue::Str(pre.clone()));
    }
    let strs =
        |xs: &[String]| MetaValue::Array(xs.iter().map(|s| MetaValue::Str(s.clone())).collect());
    if !v.tokens.is_empty() {
        m.insert("tokenizer.ggml.tokens".into(), strs(&v.tokens));
    }
    if !v.scores.is_empty() {
        m.insert(
            "tokenizer.ggml.scores".into(),
            MetaValue::Array(v.scores.iter().map(|&s| MetaValue::F32(s)).collect()),
        );
    }
    if !v.token_types.is_empty() {
        m.insert(
            "tokenizer.ggml.token_type".into(),
            MetaValue::Array(v.token_types.iter().map(|&t| MetaValue::I32(t)).collect()),
        );
    }
    if !v.merges.is_empty() {
        m.insert("tokenizer.ggml.merges".into(), strs(&v.merges));
    }
    if let Some(id) = v.bos_token_id {
        m.insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(id));
    }
    if let Some(&id) = v.eos_token_ids.first() {
        m.insert("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(id));
    }
    if let Some(&id) = v.eos_token_ids.get(1) {
        m.insert("tokenizer.ggml.eot_token_id".into(), MetaValue::U32(id));
    }
    if let Some(&id) = v.eos_token_ids.get(2) {
        m.insert("tokenizer.ggml.eom_token_id".into(), MetaValue::U32(id));
    }
    if let Some(id) = v.unk_token_id {
        m.insert("tokenizer.ggml.unknown_token_id".into(), MetaValue::U32(id));
    }
    if let Some(id) = v.pad_token_id {
        m.insert("tokenizer.ggml.padding_token_id".into(), MetaValue::U32(id));
    }
    if let Some(b) = v.add_bos_token {
        m.insert("tokenizer.ggml.add_bos_token".into(), MetaValue::Bool(b));
    }
    if let Some(b) = v.add_eos_token {
        m.insert("tokenizer.ggml.add_eos_token".into(), MetaValue::Bool(b));
    }
    if let Some(t) = &v.chat_template {
        m.insert("tokenizer.chat_template".into(), MetaValue::Str(t.clone()));
    }
    m
}

// --- rotary embedding ---------------------------------------------------------

/// `theta^(-2i/d)` for `i` in `0..d/2`: the base frequency of each rotating
/// pair. The standard RoPE table; every scaling scheme is a transform of it.
pub fn rope_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| theta.powf(-(2.0 * i as f32) / head_dim as f32))
        .collect()
}

/// The `cos` and `sin` tables for `positions`, each `positions.len() × d/2`
/// row-major, ready to be handed to a rotary kernel.
///
/// `freq_factors` is the optional per-frequency divisor a Llama 3.1+ GGUF
/// carries as `rope_freqs.weight` (llama.cpp precomputes the "llama3"
/// scaling into it); `scaling` stretches the position itself.
pub fn rope_tables(
    inv_freq: &[f32],
    freq_factors: Option<&[f32]>,
    positions: Range<usize>,
    scaling: &RopeScaling,
) -> (Vec<f32>, Vec<f32>) {
    let half = inv_freq.len();
    let n = positions.len();
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for pos in positions {
        let p = match scaling {
            RopeScaling::None => pos as f32,
            RopeScaling::Linear { factor } => pos as f32 / factor,
        };
        for (i, &f) in inv_freq.iter().enumerate() {
            let f = match freq_factors.and_then(|ff| ff.get(i)) {
                Some(&d) if d != 0.0 => f / d,
                _ => f,
            };
            let angle = p * f;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

// --- attention mask -----------------------------------------------------------

/// The additive causal mask for `t` new positions appended after `offset`
/// cached ones: `t × (offset + t)`, row-major, `0` where a query may attend
/// and `-inf` where it may not. Query `i` (absolute position `offset + i`)
/// sees every key up to and including itself.
pub fn causal_mask(t: usize, offset: usize) -> Vec<f32> {
    let l = offset + t;
    let mut m = Vec::with_capacity(t * l);
    for i in 0..t {
        for j in 0..l {
            m.push(if j <= offset + i {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    m
}

#[cfg(test)]
#[path = "logic_test.rs"]
mod tests;
