//! Model architectures on Rust tensor primitives.
//!
//! Portability: high. The domain depends on kernel and on candle (a
//! third-party tensor library, the way `serde` is a third-party
//! serialisation library). No sibling domain, no file format beyond what the
//! `deps/gguf` adapter reads.
//!
//! What this crate owns is exactly the layer infy exists for: the
//! architecture (attention, RoPE, the feed-forward block), the KV cache, and
//! how quantised weights are turned into logits. The tensor library is a
//! dependency; these are not.
//!
//! The public contract is small and deliberately tensor-free at the edge:
//! [`load`] a [`WeightSource`] into a [`Llama`], make a [`KvCache`], and call
//! [`Llama::forward`] with token ids to get a `Vec<f32>` of logits. The
//! `engine` domain never sees a `Tensor`, which is what keeps its sampling
//! and streaming logic pure and lets it be tested with no model at all.
//!
//! One architecture today (the Llama family: Llama 2/3, Mistral, SmolLM,
//! TinyLlama, and Qwen2 with its attention biases and NeoX-style RoPE). A
//! `CausalLm` trait is reserved for the second one; GUIDELINES.md says two
//! live implementations or no trait.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod cache;
pub mod deps;
mod llama;
mod logic;
mod service;

pub use cache::KvCache;
pub use deps::WeightSource;
pub use llama::Llama;
pub use logic::{
    causal_mask, rope_inv_freq, rope_tables, vocabulary_from_metadata, vocabulary_to_metadata,
    Architecture, MetaValue, Metadata, ModelConfig, RopeScaling, TensorSpec,
};
pub use service::{device, load, LoadOptions};

// Re-exported so the composition root can name a device and a dtype without
// adding candle to its own manifest for that alone.
pub use candle_core::quantized::GgmlDType;
pub use candle_core::{DType, Device};
