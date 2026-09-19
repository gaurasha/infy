//! Pure logic: the frames a streamed completion is made of, and the
//! mapping from an error to a status and an envelope.

use infy_kernel::{Error, FinishReason, Usage};
use infy_wire::{chat_chunk, code_for, error_response, ChatCompletionChunk, Delta, ErrorResponse};

/// The opening frame: the assistant role and no text, as OpenAI sends it.
pub fn first_chunk(id: &str, created: u64, model: &str) -> ChatCompletionChunk {
    chat_chunk(
        id,
        created,
        model,
        Delta {
            role: Some("assistant".into()),
            content: Some(String::new()),
        },
        None,
        None,
    )
}

/// A text frame.
pub fn delta_chunk(id: &str, created: u64, model: &str, text: &str) -> ChatCompletionChunk {
    chat_chunk(id, created, model, Delta::text(text), None, None)
}

/// The closing frame: why it stopped and what it cost.
pub fn final_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish: FinishReason,
    usage: Usage,
) -> ChatCompletionChunk {
    chat_chunk(
        id,
        created,
        model,
        Delta::default(),
        Some(finish),
        Some(usage),
    )
}

/// HTTP status and body for an error.
pub fn error_status(err: &Error) -> (u16, ErrorResponse) {
    (code_for(err).http_status(), error_response(err))
}

/// The model a request names, if it names one.
pub fn requested_model(model: Option<&str>) -> Option<&str> {
    model.map(str::trim).filter(|m| !m.is_empty())
}

#[cfg(test)]
#[path = "logic_test.rs"]
mod tests;
