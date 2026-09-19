//! Pure conversions between the protocol and kernel types, so the server and
//! the runtime client agree on every mapping and neither re-derives it.

use infy_kernel::{Completion, Error, FinishReason, Message, Result, SamplingParams, Usage};

use crate::chat::{
    ChatChoice, ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChunkChoice,
    Delta, ResponseMessage, UsageBody,
};
use crate::completion::{CompletionChoice, CompletionRequest, CompletionResponse};

/// The protocol's `finish_reason` strings. `cancelled` is an extension: OpenAI
/// has no word for it because their server never stops early.
pub fn finish_reason_str(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Stop | FinishReason::StopSequence => "stop",
        FinishReason::Length => "length",
        FinishReason::Cancelled => "cancelled",
    }
}

/// Turn a chat request into kernel messages and sampling parameters.
///
/// `defaults` supplies every parameter the client did not send, so the server
/// can be configured with a house default and a client can override per call.
pub fn chat_request_to_kernel(
    req: &ChatCompletionRequest,
    defaults: &SamplingParams,
) -> Result<(Vec<Message>, SamplingParams)> {
    if req.messages.is_empty() {
        return Err(Error::invalid("messages must not be empty"));
    }
    if let Some(n) = req.n {
        if n != 1 {
            return Err(Error::invalid(format!(
                "n must be 1, got {n}; one sequence per request"
            )));
        }
    }
    let messages = req
        .messages
        .iter()
        .map(|m| Ok(Message::new(m.role.parse()?, m.text())))
        .collect::<Result<Vec<_>>>()?;
    let params = sampling(
        defaults,
        req.max_tokens.or(req.max_completion_tokens),
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.stop.clone().map(|s| s.into_vec()),
        req.seed,
    )?;
    Ok((messages, params))
}

/// Turn a completion request into a raw prompt and sampling parameters.
pub fn completion_request_to_kernel(
    req: &CompletionRequest,
    defaults: &SamplingParams,
) -> Result<(String, SamplingParams)> {
    let params = sampling(
        defaults,
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.repetition_penalty,
        req.stop.clone().map(|s| s.into_vec()),
        req.seed,
    )?;
    Ok((req.prompt.text(), params))
}

#[allow(clippy::too_many_arguments)]
fn sampling(
    defaults: &SamplingParams,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repetition_penalty: Option<f32>,
    stop: Option<Vec<String>>,
    seed: Option<u64>,
) -> Result<SamplingParams> {
    let p = SamplingParams {
        max_tokens: max_tokens.unwrap_or(defaults.max_tokens),
        temperature: temperature.unwrap_or(defaults.temperature),
        top_p: top_p.unwrap_or(defaults.top_p),
        top_k: top_k.unwrap_or(defaults.top_k),
        repetition_penalty: repetition_penalty.unwrap_or(defaults.repetition_penalty),
        repetition_window: defaults.repetition_window,
        stop: stop.unwrap_or_else(|| defaults.stop.clone()),
        seed: seed.or(defaults.seed),
    };
    p.validate()?;
    Ok(p)
}

/// The non-streaming chat response for a finished generation.
pub fn chat_response(id: &str, created: u64, c: &Completion) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion".to_string(),
        created,
        model: c.model.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: Some(c.text.clone()),
            },
            finish_reason: Some(finish_reason_str(c.finish).to_string()),
        }],
        usage: Some(UsageBody::from(c.usage)),
    }
}

/// One streaming chat frame. The first frame carries `role`, middle frames
/// carry text, the last carries `finish_reason` and usage.
pub fn chat_chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: Delta,
    finish: Option<FinishReason>,
    usage: Option<Usage>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk".to_string(),
        created,
        model: model.to_string(),
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: finish.map(|f| finish_reason_str(f).to_string()),
        }],
        usage: usage.map(UsageBody::from),
    }
}

/// The non-streaming completion response.
pub fn completion_response(id: &str, created: u64, c: &Completion) -> CompletionResponse {
    CompletionResponse {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: c.model.clone(),
        choices: vec![CompletionChoice {
            text: c.text.clone(),
            index: 0,
            finish_reason: Some(finish_reason_str(c.finish).to_string()),
        }],
        usage: Some(UsageBody::from(c.usage)),
    }
}

/// One streaming completion frame.
pub fn completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    finish: Option<FinishReason>,
    usage: Option<Usage>,
) -> CompletionResponse {
    CompletionResponse {
        id: id.to_string(),
        object: "text_completion".to_string(),
        created,
        model: model.to_string(),
        choices: vec![CompletionChoice {
            text: text.to_string(),
            index: 0,
            finish_reason: finish.map(|f| finish_reason_str(f).to_string()),
        }],
        usage: usage.map(UsageBody::from),
    }
}

#[cfg(test)]
#[path = "convert_test.rs"]
mod tests;
