//! The imperative shell: find, start, stop, install and talk to runtimes.
//! Every decision is in `logic.rs`; this file sequences them.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use infy_kernel::{
    Cancel, Completion, Error, FinishReason, Message, Result, SamplingParams, Timing, Usage,
};
use infy_wire::{
    data_line, ChatCompletionChunk, ChatCompletionRequest, ChatMessage, CompletionChunk,
    CompletionRequest, Prompt, Stop, StreamOptions,
};

use crate::deps::{Clock, Http, PidStore, Processes};
use crate::logic::{
    expand_well_known, foreign_stop_hint, install_command, install_hint, next_step, parse_models,
    pull_command, spec, start_command, RuntimeKind, ServedModel, StartOptions, State,
};

/// What a user may override per runtime: where the binary is, and where
/// the server listens. Bootstrap values, handed in by the composition root.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeConfig {
    pub binary: Option<PathBuf>,
    pub url: Option<String>,
}

/// One runtime's state, fully described.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Status {
    pub kind: RuntimeKind,
    pub display: &'static str,
    pub state: State,
    pub binary: Option<PathBuf>,
    pub url: String,
    /// The pid infy started, if it did and it is still alive.
    pub started_by_infy: Option<u32>,
    /// What it serves, when it is running.
    pub models: Vec<ServedModel>,
    pub next_step: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Started {
    AlreadyRunning { url: String },
    Launched { pid: u32, url: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stopped {
    Killed(u32),
    NotRunning,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Installed {
    /// The install command ran and succeeded.
    Ran,
    /// There is a command; it was shown, not run.
    Command(String),
    /// No command for this OS; a page to visit.
    Manual(String),
}

/// The runtimes on this machine.
pub struct Runtimes {
    processes: Box<dyn Processes>,
    http: Box<dyn Http>,
    pids: Box<dyn PidStore>,
    clock: Box<dyn Clock>,
    overrides: HashMap<RuntimeKind, RuntimeConfig>,
    start_timeout: Duration,
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(250);

impl Runtimes {
    pub fn new(
        processes: Box<dyn Processes>,
        http: Box<dyn Http>,
        pids: Box<dyn PidStore>,
        clock: Box<dyn Clock>,
        overrides: HashMap<RuntimeKind, RuntimeConfig>,
    ) -> Self {
        Self {
            processes,
            http,
            pids,
            clock,
            overrides,
            start_timeout: Duration::from_secs(30),
        }
    }

    /// How long `start` waits for the server to answer.
    pub fn with_start_timeout(mut self, d: Duration) -> Self {
        self.start_timeout = d;
        self
    }

    /// The server URL for a runtime, without a trailing slash.
    pub fn url(&self, kind: RuntimeKind) -> String {
        self.overrides
            .get(&kind)
            .and_then(|c| c.url.clone())
            .unwrap_or_else(|| spec(kind).default_url.to_string())
            .trim_end_matches('/')
            .to_string()
    }

    /// The runtime's executable: the override if given, else `PATH`, else
    /// the places installers put it.
    pub fn binary(&self, kind: RuntimeKind) -> Option<PathBuf> {
        if let Some(b) = self.overrides.get(&kind).and_then(|c| c.binary.clone()) {
            return Some(b);
        }
        let s = spec(kind);
        for name in s.binaries {
            if let Some(p) = self.processes.which(name) {
                return Some(p);
            }
        }
        let home = self.processes.home();
        for candidate in s.well_known {
            if let Some(p) = expand_well_known(candidate, home.as_deref()) {
                if self.processes.exists(&p) {
                    return Some(p);
                }
            }
        }
        None
    }

    /// Whether the server answers.
    pub fn is_reachable(&self, kind: RuntimeKind) -> bool {
        let url = format!("{}{}", self.url(kind), spec(kind).health_path);
        self.http.get(&url, PROBE_TIMEOUT).is_ok()
    }

    fn own_pid(&self, kind: RuntimeKind) -> Result<Option<u32>> {
        match self.pids.load(kind)? {
            Some(pid) if self.processes.is_alive(pid) => Ok(Some(pid)),
            Some(_) => {
                self.pids.clear(kind)?;
                Ok(None)
            }
            None => Ok(None),
        }
    }

    pub fn status(&self, kind: RuntimeKind) -> Result<Status> {
        let s = spec(kind);
        let binary = self.binary(kind);
        let url = self.url(kind);
        let reachable = self.is_reachable(kind);
        let state = if reachable {
            State::Running
        } else if binary.is_some() {
            State::Installed
        } else {
            State::NotInstalled
        };
        let models = if reachable {
            self.models(kind).unwrap_or_default()
        } else {
            Vec::new()
        };
        Ok(Status {
            kind,
            display: s.display,
            state,
            binary,
            url,
            started_by_infy: self.own_pid(kind)?,
            models,
            next_step: next_step(kind, state, self.processes.os(), None),
        })
    }

    pub fn status_all(&self) -> Result<Vec<Status>> {
        RuntimeKind::ALL.iter().map(|&k| self.status(k)).collect()
    }

    /// Start a runtime's server and wait until it answers.
    pub fn start(&self, kind: RuntimeKind, opts: &StartOptions) -> Result<Started> {
        let s = spec(kind);
        let url = self.url(kind);
        if self.is_reachable(kind) {
            return Ok(Started::AlreadyRunning { url });
        }
        let binary = self.binary(kind).ok_or_else(|| {
            Error::unavailable(format!(
                "{} is not installed (no {} on PATH); {}",
                s.display,
                s.binaries.join("/"),
                install_hint(kind, self.processes.os())
            ))
        })?;
        let (args, env) = start_command(kind, &url, opts)?;
        let pid = self.processes.spawn_detached(&binary, &args, &env)?;
        self.pids.save(kind, pid)?;

        let deadline = self.clock.now() + self.start_timeout;
        loop {
            if self.is_reachable(kind) {
                return Ok(Started::Launched { pid, url });
            }
            if !self.processes.is_alive(pid) {
                self.pids.clear(kind)?;
                return Err(Error::unavailable(format!(
                    "{} exited right after starting; run `{} {}` by hand to see why",
                    s.display,
                    binary.display(),
                    args.join(" ")
                )));
            }
            if self.clock.now() >= deadline {
                return Err(Error::unavailable(format!(
                    "{} (pid {pid}) did not answer on {url} within {:?}; it is still running -- check its logs, or wait and retry",
                    s.display, self.start_timeout
                )));
            }
            self.processes.sleep(POLL);
        }
    }

    /// Stop a runtime infy started. One infy did not start is left alone,
    /// with the right command to stop it in the error.
    pub fn stop(&self, kind: RuntimeKind) -> Result<Stopped> {
        let s = spec(kind);
        if kind == RuntimeKind::LmStudio {
            // lms has a proper stop verb that works whoever started it.
            if let Some(bin) = self.binary(kind) {
                let _ =
                    self.processes
                        .run(&bin, &["server".into(), "stop".into()], &[], &mut |_| {});
            }
        }
        if let Some(pid) = self.own_pid(kind)? {
            self.processes.kill(pid)?;
            self.pids.clear(kind)?;
            return Ok(Stopped::Killed(pid));
        }
        if self.is_reachable(kind) && kind != RuntimeKind::LmStudio {
            return Err(Error::invalid(format!(
                "{} is running at {} but was not started by infy, so infy will not stop it; {}",
                s.display,
                self.url(kind),
                foreign_stop_hint(kind, self.processes.os())
            )));
        }
        Ok(Stopped::NotRunning)
    }

    /// Install a runtime: run the OS's command when `run` is set and one
    /// exists, otherwise say what to do.
    pub fn install(
        &self,
        kind: RuntimeKind,
        run: bool,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<Installed> {
        let os = self.processes.os();
        match install_command(kind, os) {
            Some(cmd) if run => {
                let sh = self
                    .processes
                    .which("sh")
                    .ok_or_else(|| Error::unavailable(format!("no `sh` on PATH to run: {cmd}")))?;
                let out = self
                    .processes
                    .run(&sh, &["-c".into(), cmd.into()], &[], on_line)?;
                if out.status == 0 {
                    Ok(Installed::Ran)
                } else {
                    Err(Error::unavailable(format!(
                        "install command failed with exit {}: {cmd}\n{}",
                        out.status,
                        out.stderr.lines().last().unwrap_or("")
                    )))
                }
            }
            Some(cmd) => Ok(Installed::Command(cmd.to_string())),
            None => Ok(Installed::Manual(spec(kind).install.url.to_string())),
        }
    }

    fn require_running(&self, kind: RuntimeKind) -> Result<String> {
        let url = self.url(kind);
        if self.is_reachable(kind) {
            return Ok(url);
        }
        let s = spec(kind);
        let state = if self.binary(kind).is_some() {
            State::Installed
        } else {
            State::NotInstalled
        };
        Err(Error::unavailable(format!(
            "{} is not answering at {url}; {}",
            s.display,
            next_step(kind, state, self.processes.os(), None)
        )))
    }

    /// The models a running runtime serves.
    pub fn models(&self, kind: RuntimeKind) -> Result<Vec<ServedModel>> {
        let url = self.require_running(kind)?;
        let s = spec(kind);
        let r = self
            .http
            .get(&format!("{url}{}", s.models_path), REQUEST_TIMEOUT)?;
        if r.status != 200 {
            return Err(Error::unavailable(format!(
                "{} answered {} to {}: {}",
                s.display,
                r.status,
                s.models_path,
                r.body.trim()
            )));
        }
        parse_models(kind, &r.body)
    }

    /// Fetch a model into the runtime's own store (`ollama pull`, `lms get`).
    pub fn pull(
        &self,
        kind: RuntimeKind,
        model: &str,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let args = pull_command(kind, model)?;
        let s = spec(kind);
        let bin = self.binary(kind).ok_or_else(|| {
            Error::unavailable(format!(
                "{} is not installed; {}",
                s.display,
                install_hint(kind, self.processes.os())
            ))
        })?;
        let out = self.processes.run(&bin, &args, &[], on_line)?;
        if out.status == 0 {
            Ok(())
        } else {
            Err(Error::unavailable(format!(
                "`{} {}` failed with exit {}: {}",
                bin.display(),
                args.join(" "),
                out.status,
                out.stderr
                    .lines()
                    .last()
                    .or_else(|| out.stdout.lines().last())
                    .unwrap_or("")
            )))
        }
    }

    /// Chat through a runtime's OpenAI-compatible endpoint, streaming.
    pub fn chat(
        &self,
        kind: RuntimeKind,
        cancel: &Cancel,
        model: &str,
        messages: &[Message],
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        params.validate()?;
        let url = format!("{}/v1/chat/completions", self.require_running(kind)?);
        let req = ChatCompletionRequest {
            model: Some(model.to_string()),
            messages: messages
                .iter()
                .map(|m| ChatMessage::new(m.role.as_str(), m.content.clone()))
                .collect(),
            max_tokens: Some(params.max_tokens),
            temperature: Some(params.temperature),
            top_p: Some(params.top_p),
            top_k: (params.top_k > 0).then_some(params.top_k),
            repetition_penalty: (params.repetition_penalty != 1.0)
                .then_some(params.repetition_penalty),
            stop: (!params.stop.is_empty()).then(|| Stop::Many(params.stop.clone())),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            seed: params.seed,
            ..Default::default()
        };
        let body =
            serde_json::to_string(&req).map_err(|e| Error::wrap("encoding the chat request", e))?;
        self.stream(kind, cancel, &url, body, model, sink, |payload| {
            let chunk: ChatCompletionChunk = serde_json::from_str(payload).ok()?;
            let choice = chunk.choices.into_iter().next();
            Some(Frame {
                model: chunk.model,
                text: choice.as_ref().and_then(|c| c.delta.content.clone()),
                finish: choice.and_then(|c| c.finish_reason),
                usage: chunk.usage.map(Usage::from),
            })
        })
    }

    /// Continue a raw prompt through a runtime's `/v1/completions`.
    pub fn complete(
        &self,
        kind: RuntimeKind,
        cancel: &Cancel,
        model: &str,
        prompt: &str,
        params: &SamplingParams,
        sink: &mut dyn FnMut(&str),
    ) -> Result<Completion> {
        params.validate()?;
        let url = format!("{}/v1/completions", self.require_running(kind)?);
        let req = CompletionRequest {
            model: Some(model.to_string()),
            prompt: Prompt::One(prompt.to_string()),
            max_tokens: Some(params.max_tokens),
            temperature: Some(params.temperature),
            top_p: Some(params.top_p),
            top_k: (params.top_k > 0).then_some(params.top_k),
            repetition_penalty: (params.repetition_penalty != 1.0)
                .then_some(params.repetition_penalty),
            stop: (!params.stop.is_empty()).then(|| Stop::Many(params.stop.clone())),
            stream: Some(true),
            stream_options: Some(StreamOptions {
                include_usage: Some(true),
            }),
            seed: params.seed,
        };
        let body = serde_json::to_string(&req)
            .map_err(|e| Error::wrap("encoding the completion request", e))?;
        self.stream(kind, cancel, &url, body, model, sink, |payload| {
            let chunk: CompletionChunk = serde_json::from_str(payload).ok()?;
            let choice = chunk.choices.into_iter().next();
            Some(Frame {
                model: chunk.model,
                text: choice.as_ref().map(|c| c.text.clone()),
                finish: choice.and_then(|c| c.finish_reason),
                usage: chunk.usage.map(Usage::from),
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        kind: RuntimeKind,
        cancel: &Cancel,
        url: &str,
        body: String,
        requested_model: &str,
        sink: &mut dyn FnMut(&str),
        decode: impl Fn(&str) -> Option<Frame>,
    ) -> Result<Completion> {
        let t0 = self.clock.now();
        let mut text = String::new();
        let mut answered = requested_model.to_string();
        let mut finish: Option<FinishReason> = None;
        let mut usage = Usage::default();
        let mut first: Option<Duration> = None;
        let mut leftovers = String::new();
        let status = self
            .http
            .post_json_stream(url, &body, cancel, &mut |line| {
                let Some(payload) = data_line(line) else {
                    if !line.trim().is_empty() {
                        leftovers.push_str(line.trim());
                    }
                    return;
                };
                match decode(payload) {
                    Some(frame) => {
                        if !frame.model.is_empty() {
                            answered = frame.model;
                        }
                        if let Some(t) = frame.text {
                            if !t.is_empty() {
                                first.get_or_insert_with(|| self.clock.now());
                                text.push_str(&t);
                                sink(&t);
                            }
                        }
                        if let Some(f) = frame.finish {
                            finish = Some(match f.as_str() {
                                "length" => FinishReason::Length,
                                "cancelled" => FinishReason::Cancelled,
                                _ => FinishReason::Stop,
                            });
                        }
                        if let Some(u) = frame.usage {
                            usage = u;
                        }
                    }
                    None => leftovers.push_str(payload),
                }
            })?;
        if status >= 400 {
            let s = spec(kind);
            let detail =
                infy_wire_error_message(&leftovers).unwrap_or_else(|| leftovers.trim().to_string());
            return Err(if status == 404 {
                Error::not_found(format!(
                    "model {requested_model:?} on {}: {detail}; pull it with `infy pull {}:{requested_model}` or pick one from `infy runtime models {}`",
                    s.display, kind, kind
                ))
            } else {
                Error::unavailable(format!("{} answered {status}: {detail}", s.display))
            });
        }
        let end = self.clock.now();
        let finish = if cancel.is_cancelled() {
            FinishReason::Cancelled
        } else {
            finish.unwrap_or(FinishReason::Stop)
        };
        let first = first.unwrap_or(end);
        Ok(Completion {
            model: answered,
            text,
            finish,
            usage,
            timing: Timing {
                // A runtime does not report its phases; the first token marks
                // the end of its prefill from where we stand.
                prefill: first.saturating_sub(t0),
                time_to_first_token: first.saturating_sub(t0),
                decode: end.saturating_sub(first),
                total: end.saturating_sub(t0),
            },
            // The runtime applies its own template; the request body is the
            // exact thing that was sent, and the honest answer to "what did
            // the model see" from here.
            prompt: body,
        })
    }
}

struct Frame {
    model: String,
    text: Option<String>,
    finish: Option<String>,
    usage: Option<Usage>,
}

/// The message inside an OpenAI-style error body, if that is what we got.
fn infy_wire_error_message(body: &str) -> Option<String> {
    let e: infy_wire::ErrorResponse = serde_json::from_str(body.trim()).ok()?;
    Some(e.error.message)
}

#[cfg(test)]
#[path = "service_test.rs"]
mod tests;
