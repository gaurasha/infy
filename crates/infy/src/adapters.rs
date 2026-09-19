//! The seams: where one domain's required trait is implemented over another
//! domain's concrete type. Rust has no structural typing and the orphan
//! rule forbids these impls anywhere else, so every one of them lives here,
//! and here is the only place two domains meet.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use infy_engine::{ChatRequest, Engine, GenerateRequest};
use infy_hub::{parse_ref, Source};
use infy_kernel::{Cancel, Completion, Error, Message, ModelRef, Result, SamplingParams, TokenId};
use infy_models::{KvCache, Llama};
use infy_runtime::{RuntimeKind, Runtimes};
use infy_server::ModelEntry;

// --- models → engine ------------------------------------------------------------

/// A native model as the engine sees it: tokens in, logits out.
pub struct CandleModel {
    llama: Arc<Llama>,
    id: String,
}

impl CandleModel {
    pub fn new(llama: Llama, id: String) -> Self {
        Self {
            llama: Arc::new(llama),
            id,
        }
    }
}

impl infy_engine::Model for CandleModel {
    fn id(&self) -> String {
        self.id.clone()
    }
    fn vocab_size(&self) -> usize {
        self.llama.config().vocab_size
    }
    fn context_length(&self) -> usize {
        self.llama.config().context_length
    }
    fn new_session(&self) -> Result<Box<dyn infy_engine::Session>> {
        Ok(Box::new(CandleSession {
            llama: self.llama.clone(),
            cache: self.llama.new_cache(),
        }))
    }
}

struct CandleSession {
    llama: Arc<Llama>,
    cache: KvCache,
}

impl infy_engine::Session for CandleSession {
    fn feed(&mut self, tokens: &[TokenId]) -> Result<Vec<f32>> {
        self.llama.forward(tokens, &mut self.cache)
    }
    fn len(&self) -> usize {
        self.cache.len()
    }
    fn truncate(&mut self, len: usize) {
        self.cache.truncate(len)
    }
}

// --- clocks and ids ---------------------------------------------------------------

/// One monotonic clock for every domain that wants one.
pub struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl infy_engine::Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

impl infy_runtime::Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}

impl infy_server::Clock for SystemClock {
    fn unix_seconds(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// `chatcmpl-<seconds><counter>`: unique enough for a single process.
#[derive(Default)]
pub struct SystemIds(Mutex<u64>);

impl infy_server::IdGen for SystemIds {
    fn new_id(&self, prefix: &str) -> String {
        let n = {
            let mut c = match self.0.lock() {
                Ok(c) => c,
                Err(p) => p.into_inner(),
            };
            *c += 1;
            *c
        };
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("{prefix}-{secs:x}{n:04x}")
    }
}

// --- engine / runtime → server ----------------------------------------------------

/// The server's backend: the native engine if one is loaded, plus every
/// runtime, routed by the model name a request carries.
///
/// `ollama:llama3.2` goes to ollama with the name `llama3.2`; anything
/// else, including no name at all, goes to the native engine. What
/// `/v1/models` lists is exactly what will be routed.
pub struct Dispatch {
    native: Option<(Arc<Engine>, String)>,
    runtimes: Arc<Runtimes>,
}

impl Dispatch {
    pub fn new(native: Option<Arc<Engine>>, runtimes: Arc<Runtimes>) -> Self {
        let native = native.map(|e| {
            let id = e.model_id();
            (e, id)
        });
        Self { native, runtimes }
    }

    fn route(&self, model: Option<&str>) -> Result<Route<'_>> {
        if let Some(m) = model {
            if let Ok(Source::Runtime { runtime, model }) = parse_ref(&ModelRef::new(m)?) {
                return Ok(Route::Runtime(RuntimeKind::parse(&runtime)?, model));
            }
        }
        match &self.native {
            Some((engine, _)) => Ok(Route::Native(engine)),
            None => Err(Error::not_found(format!(
                "no native model is loaded and {} names no runtime; use ollama:<name>, llamacpp:<name> or lmstudio:<name>, or start the server with --model",
                model.map(|m| format!("{m:?}")).unwrap_or_else(|| "the request".into())
            ))),
        }
    }
}

enum Route<'a> {
    Native(&'a Arc<Engine>),
    Runtime(RuntimeKind, String),
}

impl infy_server::Backend for Dispatch {
    fn models(&self) -> Vec<ModelEntry> {
        let mut out = Vec::new();
        if let Some((_, id)) = &self.native {
            out.push(ModelEntry {
                id: id.clone(),
                owned_by: "infy".into(),
            });
        }
        for kind in RuntimeKind::ALL {
            if let Ok(models) = self.runtimes.models(kind) {
                for m in models {
                    out.push(ModelEntry {
                        id: format!("{kind}:{}", m.name),
                        owned_by: kind.name().into(),
                    });
                }
            }
        }
        out
    }

    fn chat(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        messages: &[Message],
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        match self.route(model)? {
            Route::Native(engine) => engine.chat(
                cancel,
                ChatRequest {
                    messages: messages.to_vec(),
                    params: params.clone(),
                },
                sink,
            ),
            Route::Runtime(kind, name) => self
                .runtimes
                .chat(kind, cancel, &name, messages, params, sink),
        }
    }

    fn complete(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        match self.route(model)? {
            Route::Native(engine) => engine.generate(
                cancel,
                GenerateRequest {
                    prompt: prompt.to_string(),
                    params: params.clone(),
                    raw: true,
                },
                sink,
            ),
            Route::Runtime(kind, name) => self
                .runtimes
                .complete(kind, cancel, &name, prompt, params, sink),
        }
    }
}
