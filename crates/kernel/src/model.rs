use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A token id. `u32` because every tokenizer and every GGUF uses it, and a
/// vocabulary over four billion entries is not a thing.
pub type TokenId = u32;

/// Names a model the way a user does: `hf:owner/repo`, `./file.gguf`,
/// `ollama:llama3.2`. Opaque here; `hub` parses it into a source, and the
/// composition root routes it to the native engine or to a runtime.
///
/// Validated on construction so that a ref can be put in a URL, a filename or
/// a log line without a second check.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelRef(String);

impl ModelRef {
    pub fn new(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(Error::invalid("model ref is empty"));
        }
        if trimmed.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(Error::invalid(format!(
                "model ref {trimmed:?} contains whitespace or control characters"
            )));
        }
        Ok(Self(trimmed.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl FromStr for ModelRef {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::new(s)
    }
}

/// A vocabulary as a checkpoint describes it, before any tokenizer library
/// has seen it.
///
/// This is plain data lifted out of a GGUF's `tokenizer.ggml.*` metadata (or,
/// later, a `tokenizer.json`). `models` produces it; `engine` builds a
/// tokenizer from it. It lives here because both name it, and copying eight
/// fields across a newtype in the composition root would be ceremony without
/// a benefit.
///
/// `model` is the raw type name from the file -- `"llama"` (SentencePiece with
/// scores) or `"gpt2"` (byte-level BPE with merges) are the ones the engine
/// can rebuild; anything else loads its weights fine and fails at the
/// tokenizer with a message saying so. `pre` names the pre-tokenizer variant
/// (`"llama-bpe"`, `"qwen2"`, `"default"`), which selects the split regex for
/// byte-level BPE. `token_types` follow the GGUF convention: 1 normal, 2
/// unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Vocabulary {
    pub model: String,
    pub pre: Option<String>,
    pub tokens: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub bos_token_id: Option<TokenId>,
    /// Every id that ends a generation. Llama 3 has two (`<|end_of_text|>`
    /// and `<|eot_id|>`), which is why this is a list and not an option.
    pub eos_token_ids: Vec<TokenId>,
    pub unk_token_id: Option<TokenId>,
    pub pad_token_id: Option<TokenId>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    /// The Jinja chat template embedded in the checkpoint, if any.
    pub chat_template: Option<String>,
}

impl Vocabulary {
    pub fn is_eos(&self, id: TokenId) -> bool {
        self.eos_token_ids.contains(&id)
    }
}

#[cfg(test)]
#[path = "model_test.rs"]
mod tests;
