//! The imperative shell: hold the model, the tokenizer and the clock; run
//! one generation at a time; hand every decision to `logic`.

use std::sync::Mutex;
use std::time::Duration;

use infy_kernel::{
    Cancel, Completion, Error, FinishReason, Message, Result, SamplingParams, Timing, TokenId,
    Usage,
};

use crate::deps::{Clock, Model, Session, Tokenizer};
use crate::logic::{
    chatml, check_stop, needs_bos, plan_prefill, render_chat_template, sample, text_delta, Rng,
    StopCheck, CHATML_STOP,
};

/// A raw prompt to continue.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub params: SamplingParams,
    /// Send the prompt exactly as given, without a BOS even if the
    /// checkpoint wants one. What `/v1/completions` means by "raw".
    pub raw: bool,
}

/// A conversation to continue.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub params: SamplingParams,
}

struct State {
    session: Option<Box<dyn Session>>,
    /// Exactly the tokens the session has been fed, in order. Kept beside
    /// the session so the prompt-cache plan is computed on what the cache
    /// really holds, not on what we hoped it held.
    tokens: Vec<TokenId>,
}

/// The engine. `Send + Sync`; one generation runs at a time.
pub struct Engine {
    model: Box<dyn Model>,
    tokenizer: Box<dyn Tokenizer>,
    clock: Box<dyn Clock>,
    state: Mutex<State>,
}

/// After this many tokens the streaming decoder restarts its window at a
/// newline, bounding the cost of re-decoding the tail each step.
const DECODE_WINDOW: usize = 1024;

impl Engine {
    pub fn new(
        model: Box<dyn Model>,
        tokenizer: Box<dyn Tokenizer>,
        clock: Box<dyn Clock>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            clock,
            state: Mutex::new(State {
                session: None,
                tokens: Vec::new(),
            }),
        }
    }

    pub fn model_id(&self) -> String {
        self.model.id()
    }

    pub fn context_length(&self) -> usize {
        self.model.context_length()
    }

    /// The exact prompt a chat request would send, before sending it.
    pub fn render(&self, messages: &[Message]) -> Result<String> {
        match self.tokenizer.chat_template() {
            Some(t) => {
                let bos = self
                    .tokenizer
                    .bos_token()
                    .and_then(|id| self.tokenizer.special_token_text(id));
                let eos = self
                    .tokenizer
                    .eos_tokens()
                    .first()
                    .and_then(|&id| self.tokenizer.special_token_text(id));
                render_chat_template(&t, messages, bos.as_deref(), eos.as_deref())
            }
            None => Ok(chatml(messages)),
        }
    }

    /// Drop the cached session. The next call prefills from scratch.
    pub fn reset_cache(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.session = None;
            s.tokens.clear();
        }
    }

    /// Continue a conversation: render, then [`generate`](Self::generate).
    pub fn chat(
        &self,
        cancel: &Cancel,
        req: ChatRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        if req.messages.is_empty() {
            return Err(Error::invalid("chat request has no messages"));
        }
        let prompt = self.render(&req.messages)?;
        let mut params = req.params;
        if self.tokenizer.chat_template().is_none() && !params.stop.iter().any(|s| s == CHATML_STOP)
        {
            params.stop.push(CHATML_STOP.to_string());
        }
        self.generate(
            cancel,
            GenerateRequest {
                prompt,
                params,
                raw: false,
            },
            sink,
        )
    }

    /// Continue a prompt, streaming text to `sink` as it becomes complete.
    pub fn generate(
        &self,
        cancel: &Cancel,
        req: GenerateRequest,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        req.params.validate()?;
        let t0 = self.clock.now();
        let params = &req.params;

        // --- tokenise ---
        let mut prompt_tokens = self.tokenizer.encode(&req.prompt)?;
        let bos_text = self
            .tokenizer
            .bos_token()
            .and_then(|id| self.tokenizer.special_token_text(id));
        if !req.raw && needs_bos(&req.prompt, bos_text.as_deref(), self.tokenizer.add_bos()) {
            if let Some(bos) = self.tokenizer.bos_token() {
                prompt_tokens.insert(0, bos);
            }
        }
        if prompt_tokens.is_empty() {
            return Err(Error::invalid("prompt is empty"));
        }
        let ctx = self.model.context_length();
        if prompt_tokens.len() >= ctx {
            return Err(Error::invalid(format!(
                "prompt is {} tokens but the context window is {ctx}; leave room for at least one generated token",
                prompt_tokens.len()
            )));
        }
        let max_new = params.max_tokens.min(ctx - prompt_tokens.len());
        let eos = self.tokenizer.eos_tokens();

        // --- prefill, reusing the cache where the prompt matches ---
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::internal("engine state poisoned"))?;
        if state.session.is_none() {
            state.session = Some(self.model.new_session()?);
            state.tokens.clear();
        }
        let plan = plan_prefill(&state.tokens, &prompt_tokens);
        state.tokens.truncate(plan.keep);
        let State {
            session,
            tokens: fed,
        } = &mut *state;
        let session = session
            .as_mut()
            .ok_or_else(|| Error::internal("session vanished"))?;
        session.truncate(plan.keep);
        cancel.check()?;
        let mut logits = match session.feed(&prompt_tokens[plan.feed_from..]) {
            Ok(l) => l,
            Err(e) => {
                // A failed feed leaves the cache in an unknown state; start
                // the next call from nothing rather than from a guess.
                state.session = None;
                state.tokens.clear();
                return Err(e);
            }
        };
        fed.extend_from_slice(&prompt_tokens[plan.feed_from..]);
        let t_prefill = self.clock.now();

        // --- decode ---
        let seed = params
            .seed
            .unwrap_or(t0.as_nanos() as u64 ^ 0xA076_1D64_78BD_642F);
        let mut rng = Rng::new(seed);
        let mut all = prompt_tokens.clone();
        let mut generated: Vec<TokenId> = Vec::new();
        let mut window_start = 0usize;
        let mut prev_text = String::new();
        let mut pending = String::new();
        let mut text = String::new();
        let mut finish = FinishReason::Length;
        let mut t_first: Option<Duration> = None;
        let mut emit = |s: &str, text: &mut String| {
            if !s.is_empty() {
                text.push_str(s);
                sink(s);
            }
        };

        for step in 0..max_new {
            if cancel.is_cancelled() {
                finish = FinishReason::Cancelled;
                break;
            }
            let next = sample(&logits, params, &all, rng.next_f32());
            if step == 0 {
                t_first = Some(self.clock.now());
            }
            if eos.contains(&next) {
                finish = FinishReason::Stop;
                break;
            }
            generated.push(next);
            all.push(next);

            let cur = self.tokenizer.decode(&generated[window_start..])?;
            if let Some(delta) = text_delta(&prev_text, &cur) {
                pending.push_str(delta);
                prev_text = cur;
                if generated.len() - window_start > DECODE_WINDOW && prev_text.ends_with('\n') {
                    window_start = generated.len();
                    prev_text.clear();
                }
            }
            match check_stop(&pending, &params.stop) {
                StopCheck::Hit(i) => {
                    let head = pending[..i].to_string();
                    emit(&head, &mut text);
                    pending.clear();
                    finish = FinishReason::StopSequence;
                    break;
                }
                StopCheck::Partial(hold) => {
                    let split = pending.len() - hold;
                    let head = pending[..split].to_string();
                    emit(&head, &mut text);
                    pending.drain(..split);
                }
                StopCheck::NoMatch => {
                    let head = std::mem::take(&mut pending);
                    emit(&head, &mut text);
                }
            }

            if step + 1 == max_new {
                finish = FinishReason::Length;
                break;
            }
            logits = match session.feed(&[next]) {
                Ok(l) => l,
                Err(e) => {
                    state.session = None;
                    state.tokens.clear();
                    return Err(e);
                }
            };
            fed.push(next);
        }
        if finish != FinishReason::StopSequence {
            let tail = std::mem::take(&mut pending);
            emit(&tail, &mut text);
        }

        let t_end = self.clock.now();
        let ttft = t_first.unwrap_or(t_end).saturating_sub(t0);
        Ok(Completion {
            model: self.model.id(),
            text,
            finish,
            usage: Usage {
                prompt_tokens: prompt_tokens.len(),
                completion_tokens: generated.len(),
                cached_tokens: plan.keep,
            },
            timing: Timing {
                prefill: t_prefill.saturating_sub(t0),
                time_to_first_token: ttft,
                decode: t_end.saturating_sub(t_first.unwrap_or(t_end)),
                total: t_end.saturating_sub(t0),
            },
            prompt: req.prompt,
        })
    }
}

#[cfg(test)]
#[path = "service_test.rs"]
mod tests;
