use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// How to turn logits into tokens, and when to stop.
///
/// `temperature == 0.0` is greedy. `top_k == 0` and `top_p >= 1.0` disable
/// those filters. `repetition_penalty == 1.0` disables the penalty. The
/// defaults are the ones that make a small instruct model pleasant to talk to
/// without any flags.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// How many of the most recent tokens the penalty looks at.
    pub repetition_window: usize,
    pub stop: Vec<String>,
    /// A seed makes a run reproducible. `None` means the caller does not care,
    /// and the composition root picks one from the clock.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            repetition_window: 64,
            stop: Vec::new(),
            seed: None,
        }
    }
}

impl SamplingParams {
    /// Greedy decoding: the most likely token every step. What a test wants.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            ..Self::default()
        }
    }

    /// Pure validation, so the wire layer and the CLI reject the same inputs
    /// with the same message.
    pub fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 {
            return Err(Error::invalid("max_tokens must be at least 1"));
        }
        if self.temperature < 0.0 || !self.temperature.is_finite() {
            return Err(Error::invalid(format!(
                "temperature must be a finite number >= 0, got {}",
                self.temperature
            )));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(Error::invalid(format!(
                "top_p must be in (0, 1], got {}",
                self.top_p
            )));
        }
        if self.repetition_penalty <= 0.0 || !self.repetition_penalty.is_finite() {
            return Err(Error::invalid(format!(
                "repetition_penalty must be a finite number > 0, got {}",
                self.repetition_penalty
            )));
        }
        if self.stop.iter().any(|s| s.is_empty()) {
            return Err(Error::invalid("stop sequences must not be empty strings"));
        }
        Ok(())
    }
}

/// Why a generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Stop,
    /// `max_tokens` was reached, or the context window filled.
    Length,
    /// One of the caller's stop sequences appeared.
    StopSequence,
    /// The caller cancelled. The partial text is kept.
    Cancelled,
}

/// Token accounting for one generation.
///
/// Recorded on every call, because every optimisation in this repository is
/// judged by these numbers and a benchmark someone remembers to run is not
/// the same as a measurement taken every time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// The part of the prompt served from a cache rather than recomputed.
    pub cached_tokens: usize,
}

impl Usage {
    pub fn total(&self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }
}

/// Where the time went.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timing {
    /// Processing the prompt tokens that were not cached.
    pub prefill: Duration,
    /// From the start of the call to the first emitted token.
    pub time_to_first_token: Duration,
    /// Generating every token after the first.
    pub decode: Duration,
    pub total: Duration,
}

impl Timing {
    /// Tokens per second over the decode phase, the number people quote.
    /// Zero when nothing was decoded, rather than a division by zero.
    pub fn decode_tokens_per_second(&self, completion_tokens: usize) -> f64 {
        rate(completion_tokens.saturating_sub(1), self.decode)
    }

    /// Prompt tokens per second over the prefill phase.
    pub fn prefill_tokens_per_second(&self, fed_tokens: usize) -> f64 {
        rate(fed_tokens, self.prefill)
    }
}

fn rate(count: usize, d: Duration) -> f64 {
    let secs = d.as_secs_f64();
    if count == 0 || secs <= 0.0 {
        0.0
    } else {
        count as f64 / secs
    }
}

/// The terminal result of a generation: the text, why it stopped, what it
/// cost, and the exact prompt that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Completion {
    /// What actually answered, which is not always what was asked for.
    pub model: String,
    /// The assembled text, so a caller that ignored the stream still has it.
    pub text: String,
    pub finish: FinishReason,
    pub usage: Usage,
    pub timing: Timing,
    /// The prompt exactly as sent, after the chat template, as a string. When
    /// an answer is bad this is the only question worth asking.
    pub prompt: String,
}

#[cfg(test)]
#[path = "generation_test.rs"]
mod tests;
