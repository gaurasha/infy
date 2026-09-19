use infy_kernel::{Cancel, Completion, Message, Result, SamplingParams};

/// A model the server offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    /// Who serves it: `infy` for the native engine, a runtime's name otherwise.
    pub owned_by: String,
}

/// What the server needs from whatever generates text. The composition
/// root implements it over the native engine, over a runtime, or over a
/// dispatcher that picks by model name; the server cannot tell and must
/// not care.
///
/// `model` is what the client asked for, or `None`. The backend decides
/// what that means -- a native server may serve any name with its one
/// model, a dispatcher may route on it -- and reports what actually
/// answered in the completion.
pub trait Backend: Send + Sync {
    fn models(&self) -> Vec<ModelEntry>;

    fn chat(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        messages: &[Message],
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion>;

    fn complete(
        &self,
        cancel: &Cancel,
        model: Option<&str>,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion>;
}

/// Response ids like `chatcmpl-...`. Injected so tests see stable ids.
pub trait IdGen: Send + Sync {
    fn new_id(&self, prefix: &str) -> String;
}

/// Wall-clock seconds for the `created` field.
pub trait Clock: Send + Sync {
    fn unix_seconds(&self) -> u64;
}

pub mod memory;
