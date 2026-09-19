//! The types every layer of infy names, and nothing else.
//!
//! Portability: total. `infy-kernel` depends on `serde` and the standard
//! library, and is the one crate every domain is permitted to depend on. That
//! permission is what makes the no-sibling-dependencies rule in GUIDELINES.md
//! possible to keep at all.
//!
//! The constraint is strict and load-bearing: kernel contains types and pure
//! functions. It must never open a file, spawn a thread, take a tensor type or
//! acquire another dependency. The moment it does, every domain in the system
//! inherits a transitive dependency on whatever it touched, and the isolation
//! the rest of the architecture buys is gone.
//!
//! Two consequences worth spelling out. [`Cancel`] is a flag, not a runtime:
//! it is here because every long-running operation takes one, and it is an
//! atomic rather than a channel so that it needs nothing beyond `std`.
//! [`Vocabulary`] is here because two domains name it -- `models` reads it out
//! of a GGUF and `engine` builds a tokenizer from it -- and the alternative is
//! copying eight fields across a newtype in the composition root.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod cancel;
mod chat;
mod error;
mod generation;
mod model;

pub use cancel::Cancel;
pub use chat::{Message, Role};
pub use error::{Error, Result};
pub use generation::{Completion, FinishReason, SamplingParams, Timing, Usage};
pub use model::{ModelRef, TokenId, Vocabulary};
