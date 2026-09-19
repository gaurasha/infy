//! In-memory implementations of the server's dependencies: a backend that
//! answers with a canned reply word by word, counting ids, a fixed clock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use infy_kernel::{
    Cancel, Completion, Error, FinishReason, Message, Result, SamplingParams, Timing, Usage,
};

use crate::deps::{Backend, Clock, IdGen, ModelEntry};

/// A recorded call.
#[derive(Debug, Clone, PartialEq)]
pub struct Call {
    pub model: Option<String>,
    pub prompt: String,
    pub params: SamplingParams,
}

/// Streams `reply` one word at a time. Asking for the model `missing`
/// is a `NotFound`; `explode` fails after the first word.
pub struct MemoryBackend {
    reply: String,
    calls: Mutex<Vec<Call>>,
    /// Set to make the backend block until cancelled, for disconnect tests.
    pub hang_until_cancelled: bool,
}

impl MemoryBackend {
    pub fn new(reply: &str) -> Self {
        Self {
            reply: reply.to_string(),
            calls: Mutex::new(Vec::new()),
            hang_until_cancelled: false,
        }
    }

    pub fn calls(&self) -> Vec<Call> {
        self.calls.lock().map(|c| c.clone()).unwrap_or_default()
    }

    fn answer(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        prompt: String,
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        if let Ok(mut c) = self.calls.lock() {
            c.push(Call {
                model: model.map(String::from),
                prompt: prompt.clone(),
                params: params.clone(),
            });
        }
        if model == Some("missing") {
            return Err(Error::not_found(format!("model {:?}", "missing")));
        }
        if self.hang_until_cancelled {
            while !cancel.is_cancelled() {
                std::thread::sleep(Duration::from_millis(5));
            }
            return Err(Error::Cancelled);
        }
        let mut text = String::new();
        let words: Vec<&str> = self.reply.split_inclusive(' ').collect();
        for (i, w) in words.iter().enumerate() {
            if cancel.is_cancelled() {
                break;
            }
            if model == Some("explode") && i == 1 {
                return Err(Error::unavailable("the backend fell over"));
            }
            text.push_str(w);
            sink(w);
        }
        let finish = if cancel.is_cancelled() {
            FinishReason::Cancelled
        } else {
            FinishReason::Stop
        };
        Ok(Completion {
            model: model.unwrap_or("memory").to_string(),
            text,
            finish,
            usage: Usage {
                prompt_tokens: prompt.len(),
                completion_tokens: words.len(),
                cached_tokens: 0,
            },
            timing: Timing::default(),
            prompt,
        })
    }
}

impl Backend for MemoryBackend {
    fn models(&self) -> Vec<ModelEntry> {
        vec![ModelEntry {
            id: "memory".into(),
            owned_by: "infy".into(),
        }]
    }

    fn chat(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        messages: &[Message],
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        let prompt = messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");
        self.answer(cancel, model, prompt, params, sink)
    }

    fn complete(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        self.answer(cancel, model, prompt.to_string(), params, sink)
    }
}

/// `prefix-1`, `prefix-2`, ...
#[derive(Default)]
pub struct MemoryIds(AtomicU64);

impl IdGen for MemoryIds {
    fn new_id(&self, prefix: &str) -> String {
        format!("{prefix}-{}", self.0.fetch_add(1, Ordering::SeqCst) + 1)
    }
}

pub struct MemoryClock(pub u64);

impl Clock for MemoryClock {
    fn unix_seconds(&self) -> u64 {
        self.0
    }
}
