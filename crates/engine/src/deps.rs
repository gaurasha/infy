use std::time::Duration;

use infy_kernel::{Result, TokenId};

/// A language model as the engine sees it: a vocabulary size, a context
/// length, and sessions that turn tokens into logits.
///
/// Two live implementations from day one -- `deps/memory.rs` and, through a
/// newtype in the composition root, the `models` crate -- which is the bar
/// GUIDELINES.md sets for defining a trait at all.
pub trait Model: Send + Sync {
    /// What answers, for the `model` field of every completion.
    fn id(&self) -> String;
    fn vocab_size(&self) -> usize;
    fn context_length(&self) -> usize;
    /// A fresh sequence state with an empty cache.
    fn new_session(&self) -> Result<Box<dyn Session>>;
}

/// One sequence's state. Feeding tokens appends to it; the returned logits
/// are for the last token fed.
pub trait Session: Send {
    /// Append `tokens` and return one logit per vocabulary entry for the
    /// next position.
    fn feed(&mut self, tokens: &[TokenId]) -> Result<Vec<f32>>;
    /// Positions currently held.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Forget every position from `len` on, so a prompt sharing a prefix
    /// with the previous one only has to feed its tail.
    fn truncate(&mut self, len: usize);
}

/// Text ↔ tokens. `encode` never adds special tokens; the engine decides
/// about BOS itself so the rule is in one place and tested.
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;
    /// Decode, skipping special tokens. A trailing incomplete UTF-8 sequence
    /// must decode to U+FFFD, which is how the engine knows to wait.
    fn decode(&self, tokens: &[TokenId]) -> Result<String>;
    fn bos_token(&self) -> Option<TokenId>;
    /// Whether this checkpoint expects a BOS at the start of every prompt.
    fn add_bos(&self) -> bool;
    /// Every id that ends a generation.
    fn eos_tokens(&self) -> Vec<TokenId>;
    /// The checkpoint's own Jinja chat template, if it has one.
    fn chat_template(&self) -> Option<String>;
    /// The text of a special token, for templates that splice `bos_token`
    /// and `eos_token` in as strings.
    fn special_token_text(&self, id: TokenId) -> Option<String>;
}

/// A monotonic clock, injected so timing is testable. The epoch is
/// arbitrary; only differences are meaningful.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

pub mod memory;
#[cfg(feature = "tokenizers")]
pub mod tokenizers;
