//! The OpenAI-compatible HTTP API.
//!
//! Portability: high. Depends on kernel, wire, axum and tokio. No sibling
//! domain: it does not know whether a model runs natively or through
//! ollama, only that something implementing [`deps::Backend`] answers.
//!
//! Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`
//! and `POST /v1/completions`, streaming and not. Any client that speaks to
//! OpenAI speaks to this.
//!
//! The backend is synchronous and generates one sequence at a time; the
//! server runs each call on a blocking thread and, when streaming, forwards
//! text through a channel into server-sent events. When the client goes
//! away the stream is dropped, and dropping it cancels the generation --
//! nobody is listening, so nothing is computed.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod deps;
mod logic;
mod service;

pub use deps::{Backend, Clock, IdGen, ModelEntry};
pub use logic::{delta_chunk, error_status, final_chunk, first_chunk, requested_model};
pub use service::{router, serve, Server};
