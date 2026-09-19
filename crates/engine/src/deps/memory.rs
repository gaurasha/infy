//! In-memory implementations of every trait the engine requires, shipped
//! with the domain so its tests need no weights, no tokenizer file and no
//! clock. They are real implementations, not stubs: the byte tokenizer
//! splits multi-byte characters exactly the way a byte-level BPE does, and
//! the model produces logits, not tokens.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use infy_kernel::{Error, Result, TokenId};

use crate::deps::{Clock, Model, Session, Tokenizer};

/// What comes next, given everything fed so far.
pub type Rule = Arc<dyn Fn(&[TokenId]) -> TokenId + Send + Sync>;

/// A deterministic model: a rule maps everything fed so far to the token
/// that should come next, and the logits put that token far above the rest.
pub struct MemoryModel {
    id: String,
    vocab_size: usize,
    context_length: usize,
    rule: Rule,
    feeds: Arc<Mutex<Vec<Vec<TokenId>>>>,
}

impl MemoryModel {
    pub fn new(
        vocab_size: usize,
        context_length: usize,
        rule: impl Fn(&[TokenId]) -> TokenId + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: "memory".into(),
            vocab_size,
            context_length,
            rule: Arc::new(rule),
            feeds: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Answers with the bytes of `reply` and then EOS, whatever the prompt.
    /// It finds its place by matching the longest suffix of the sequence
    /// against the reply, so it works across turns and cache reuse alike.
    pub fn reply(reply: &str) -> Self {
        let bytes: Vec<TokenId> = reply.bytes().map(TokenId::from).collect();
        Self::new(ByteTokenizer::VOCAB, 4096, move |seq| {
            let n = progress(seq, &bytes);
            bytes.get(n).copied().unwrap_or(ByteTokenizer::EOS)
        })
    }

    /// Emits the bytes of `text` over and over, never an EOS: for tests of
    /// `max_tokens`, stop sequences and cancellation.
    pub fn cycle(text: &str) -> Self {
        let bytes: Vec<TokenId> = text.bytes().map(TokenId::from).collect();
        Self::new(ByteTokenizer::VOCAB, 4096, move |seq| {
            let n = progress(seq, &bytes);
            bytes[n % bytes.len()]
        })
    }

    pub fn with_context_length(mut self, n: usize) -> Self {
        self.context_length = n;
        self
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = id.into();
        self
    }

    /// Every slice of tokens any session was fed, in order: what a test of
    /// the prompt cache inspects.
    pub fn feeds(&self) -> Vec<Vec<TokenId>> {
        self.feeds.lock().map(|f| f.clone()).unwrap_or_default()
    }

    /// A handle to the same log, for a test that hands the model to an
    /// engine and inspects the feeds afterwards.
    pub fn feed_log(&self) -> Arc<Mutex<Vec<Vec<TokenId>>>> {
        self.feeds.clone()
    }
}

/// How far into `pattern` the sequence's tail has got, cyclically.
fn progress(seq: &[TokenId], pattern: &[TokenId]) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    let max = seq.len().min(pattern.len());
    for k in (1..=max).rev() {
        if seq[seq.len() - k..] == pattern[..k] {
            return k;
        }
    }
    0
}

impl Model for MemoryModel {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn context_length(&self) -> usize {
        self.context_length
    }
    fn new_session(&self) -> Result<Box<dyn Session>> {
        Ok(Box::new(MemorySession {
            tokens: Vec::new(),
            vocab_size: self.vocab_size,
            context_length: self.context_length,
            rule: self.rule.clone(),
            feeds: self.feeds.clone(),
        }))
    }
}

struct MemorySession {
    tokens: Vec<TokenId>,
    vocab_size: usize,
    context_length: usize,
    rule: Rule,
    feeds: Arc<Mutex<Vec<Vec<TokenId>>>>,
}

impl Session for MemorySession {
    fn feed(&mut self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(Error::invalid("feed called with no tokens"));
        }
        if self.tokens.len() + tokens.len() > self.context_length {
            return Err(Error::invalid(format!(
                "{} tokens exceed the context length of {}",
                self.tokens.len() + tokens.len(),
                self.context_length
            )));
        }
        if let Ok(mut f) = self.feeds.lock() {
            f.push(tokens.to_vec());
        }
        self.tokens.extend_from_slice(tokens);
        let next = (self.rule)(&self.tokens) as usize;
        let mut logits = vec![-50.0; self.vocab_size];
        if let Some(l) = logits.get_mut(next) {
            *l = 50.0;
        }
        Ok(logits)
    }

    fn len(&self) -> usize {
        self.tokens.len()
    }

    fn truncate(&mut self, len: usize) {
        self.tokens.truncate(len);
    }
}

/// Bytes are tokens: id `b` is byte `b`, `256` is BOS, `257` is EOS. Decoding
/// an incomplete multi-byte character yields U+FFFD, exactly like a real
/// byte-level tokenizer, so the engine's streaming path is exercised for
/// real.
pub struct ByteTokenizer {
    add_bos: bool,
    chat_template: Option<String>,
}

impl ByteTokenizer {
    pub const BOS: TokenId = 256;
    pub const EOS: TokenId = 257;
    pub const VOCAB: usize = 258;

    pub fn new() -> Self {
        Self {
            add_bos: false,
            chat_template: None,
        }
    }

    pub fn with_add_bos(mut self, yes: bool) -> Self {
        self.add_bos = yes;
        self
    }

    pub fn with_chat_template(mut self, t: &str) -> Self {
        self.chat_template = Some(t.to_string());
        self
    }
}

impl Default for ByteTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        Ok(text.bytes().map(TokenId::from).collect())
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter(|&&t| t < 256)
            .map(|&t| t as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn bos_token(&self) -> Option<TokenId> {
        Some(Self::BOS)
    }

    fn add_bos(&self) -> bool {
        self.add_bos
    }

    fn eos_tokens(&self) -> Vec<TokenId> {
        vec![Self::EOS]
    }

    fn chat_template(&self) -> Option<String> {
        self.chat_template.clone()
    }

    fn special_token_text(&self, id: TokenId) -> Option<String> {
        match id {
            Self::BOS => Some("<s>".into()),
            Self::EOS => Some("</s>".into()),
            _ => None,
        }
    }
}

/// A clock that advances a fixed step on every read, so timings are
/// deterministic and every phase measurably non-zero.
pub struct MemoryClock {
    now: Mutex<Duration>,
    step: Duration,
}

impl MemoryClock {
    pub fn new(step: Duration) -> Self {
        Self {
            now: Mutex::new(Duration::ZERO),
            step,
        }
    }
}

impl Default for MemoryClock {
    fn default() -> Self {
        Self::new(Duration::from_millis(1))
    }
}

impl Clock for MemoryClock {
    fn now(&self) -> Duration {
        let mut n = match self.now.lock() {
            Ok(n) => n,
            Err(p) => p.into_inner(),
        };
        *n += self.step;
        *n
    }
}
