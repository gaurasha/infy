//! The OpenAI-compatible chat-completions protocol: request, response and
//! streaming chunk types, error codes, and SSE framing.
//!
//! Portability: total. `infy-wire` depends on `infy-kernel` and `serde`,
//! nothing else. Like kernel it holds types and pure functions only.
//!
//! It is a shared leaf rather than part of the `server` domain because it is
//! spoken in BOTH directions: our server serves it, and the `runtime` domain
//! speaks it as a client to ollama, llama.cpp and LM Studio, which all expose
//! the same protocol. One set of types, one set of tests, and a runtime's
//! streaming response is decoded by the same code that encodes ours.
//!
//! The types are deliberately permissive on input (`Option` everywhere a
//! client might omit a field, unknown fields ignored) and exact on output.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod chat;
mod completion;
mod convert;
mod error;
mod models;
mod sse;

pub use chat::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    ChunkChoice, Content, ContentPart, Delta, ResponseMessage, Stop, StreamOptions, UsageBody,
};
pub use completion::{
    CompletionChoice, CompletionChunk, CompletionRequest, CompletionResponse, Prompt,
};
pub use convert::{
    chat_chunk, chat_request_to_kernel, chat_response, completion_chunk,
    completion_request_to_kernel, completion_response, finish_reason_str,
};
pub use error::{code_for, error_response, Code, ErrorBody, ErrorResponse};
pub use models::{ModelList, ModelObject};
pub use sse::{data_line, frame, DONE_FRAME};
