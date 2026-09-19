//! The native generation loop.
//!
//! Portability: high. The domain depends on kernel, on `minijinja` for chat
//! templates, and -- behind the `tokenizers` feature -- on the `tokenizers`
//! crate for its real tokenizer adapter. No sibling domain: it never sees a
//! tensor, a file, or a network.
//!
//! The boundary is "tokens in, logits out". [`deps::Model`] hands the engine
//! a `Vec<f32>` per step and nothing else, so sampling, stop sequences,
//! UTF-8-safe streaming and the prompt cache are pure functions in
//! `logic.rs`, tested by tables, and identical whether the logits came from
//! a GPU or from the in-memory test model in `deps/memory.rs`.
//!
//! What the engine does per call, in order: render the chat template if
//! there is one, tokenise, decide how much of the previous session's cache
//! can be kept (the prompt cache), feed the rest, then sample one token at a
//! time, decoding text as soon as it is complete and holding back anything
//! that might be the start of a stop sequence. Every call returns a
//! [`Completion`](infy_kernel::Completion) with usage, timing, and the exact
//! prompt as sent.
//!
//! One generation at a time: the engine holds one session under a mutex and
//! callers queue. Batching is a roadmap item, not a mutex to remove.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod deps;
mod logic;
mod service;

pub use deps::{Clock, Model, Session, Tokenizer};
pub use logic::{
    chatml, check_stop, needs_bos, plan_prefill, render_chat_template, sample, text_delta,
    PrefillPlan, Rng, StopCheck,
};
pub use service::{ChatRequest, Engine, GenerateRequest};
