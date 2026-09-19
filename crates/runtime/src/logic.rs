//! Pure logic: the runtime specs, the decisions made from what was found,
//! and the parsing of what runtimes answer.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use infy_kernel::{Error, Result};

use crate::deps::Os;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeKind {
    Ollama,
    LlamaCpp,
    LmStudio,
}

impl RuntimeKind {
    pub const ALL: [RuntimeKind; 3] = [
        RuntimeKind::Ollama,
        RuntimeKind::LlamaCpp,
        RuntimeKind::LmStudio,
    ];

    /// The name used in refs (`ollama:`) and on the command line.
    pub fn name(self) -> &'static str {
        match self {
            RuntimeKind::Ollama => "ollama",
            RuntimeKind::LlamaCpp => "llamacpp",
            RuntimeKind::LmStudio => "lmstudio",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "ollama" => Ok(RuntimeKind::Ollama),
            "llamacpp" | "llama.cpp" | "llama-cpp" | "llama-server" => Ok(RuntimeKind::LlamaCpp),
            "lmstudio" | "lm-studio" | "lms" => Ok(RuntimeKind::LmStudio),
            other => Err(Error::invalid(format!(
                "unknown runtime {other:?}; infy knows ollama, llamacpp and lmstudio"
            ))),
        }
    }
}

impl std::fmt::Display for RuntimeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// How to install a runtime on each OS: a command where one exists, and
/// the page to visit where it does not.
#[derive(Debug, Clone, Copy)]
pub struct Install {
    pub linux: Option<&'static str>,
    pub macos: Option<&'static str>,
    pub windows: Option<&'static str>,
    pub url: &'static str,
}

/// Everything that differs between runtimes. The rest is code, written once.
#[derive(Debug, Clone, Copy)]
pub struct RuntimeSpec {
    pub kind: RuntimeKind,
    pub display: &'static str,
    /// Executable names to look for on `PATH`.
    pub binaries: &'static [&'static str],
    /// Places installers put it that are often not on `PATH`. `~` is the home.
    pub well_known: &'static [&'static str],
    pub default_url: &'static str,
    /// A path that answers when the server is up.
    pub health_path: &'static str,
    /// Where the served models are listed.
    pub models_path: &'static str,
    /// Whether starting needs a model file on the command line.
    pub needs_model_to_start: bool,
    pub install: Install,
}

pub const SPECS: [RuntimeSpec; 3] = [
    RuntimeSpec {
        kind: RuntimeKind::Ollama,
        display: "ollama",
        binaries: &["ollama"],
        well_known: &[
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama",
            "~/.ollama/bin/ollama",
            "~/AppData/Local/Programs/Ollama/ollama.exe",
        ],
        default_url: "http://127.0.0.1:11434",
        health_path: "/",
        models_path: "/api/tags",
        needs_model_to_start: false,
        install: Install {
            linux: Some("curl -fsSL https://ollama.com/install.sh | sh"),
            macos: Some("brew install ollama"),
            windows: None,
            url: "https://ollama.com/download",
        },
    },
    RuntimeSpec {
        kind: RuntimeKind::LlamaCpp,
        display: "llama.cpp",
        binaries: &["llama-server"],
        well_known: &[
            "/opt/homebrew/bin/llama-server",
            "/usr/local/bin/llama-server",
            "~/llama.cpp/build/bin/llama-server",
            "~/.local/bin/llama-server",
        ],
        default_url: "http://127.0.0.1:8080",
        health_path: "/health",
        models_path: "/v1/models",
        needs_model_to_start: true,
        install: Install {
            linux: None,
            macos: Some("brew install llama.cpp"),
            windows: None,
            url: "https://github.com/ggml-org/llama.cpp/releases",
        },
    },
    RuntimeSpec {
        kind: RuntimeKind::LmStudio,
        display: "LM Studio",
        binaries: &["lms"],
        well_known: &[
            "~/.lmstudio/bin/lms",
            "~/.cache/lm-studio/bin/lms",
            "~/.lmstudio/bin/lms.exe",
        ],
        default_url: "http://127.0.0.1:1234",
        health_path: "/v1/models",
        models_path: "/v1/models",
        needs_model_to_start: false,
        install: Install {
            linux: None,
            macos: None,
            windows: None,
            url: "https://lmstudio.ai/download",
        },
    },
];

pub fn spec(kind: RuntimeKind) -> &'static RuntimeSpec {
    match kind {
        RuntimeKind::Ollama => &SPECS[0],
        RuntimeKind::LlamaCpp => &SPECS[1],
        RuntimeKind::LmStudio => &SPECS[2],
    }
}

/// `~/x` with the home directory filled in.
pub fn expand_well_known(path: &str, home: Option<&Path>) -> Option<PathBuf> {
    match path.strip_prefix("~/") {
        Some(rest) => home.map(|h| h.join(rest)),
        None => Some(PathBuf::from(path)),
    }
}

/// What was found, summarised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// No binary and no server answering.
    NotInstalled,
    /// A binary exists; nothing is listening.
    Installed,
    /// The server answers.
    Running,
}

/// The one-line next step for a state, always concrete.
pub fn next_step(kind: RuntimeKind, state: State, os: Os, model_hint: Option<&str>) -> String {
    let s = spec(kind);
    match state {
        State::NotInstalled => format!("install it: {}", install_hint(kind, os)),
        State::Installed => match kind {
            RuntimeKind::LlamaCpp => format!(
                "start it with a model: infy runtime start llamacpp --model {}",
                model_hint.unwrap_or("<file.gguf>")
            ),
            _ => format!("start it: infy runtime start {}", s.kind),
        },
        State::Running => format!("use a model: infy chat --model {}:<name>", s.kind),
    }
}

/// What to run to install, if there is a command for this OS.
pub fn install_command(kind: RuntimeKind, os: Os) -> Option<&'static str> {
    let i = spec(kind).install;
    match os {
        Os::Linux => i.linux,
        Os::MacOs => i.macos,
        Os::Windows => i.windows,
    }
}

/// The install instruction for a person: the command where there is one,
/// the download page otherwise.
pub fn install_hint(kind: RuntimeKind, os: Os) -> String {
    match install_command(kind, os) {
        Some(cmd) => format!("`{cmd}` (or download from {})", spec(kind).install.url),
        None => format!("download from {}", spec(kind).install.url),
    }
}

/// What a start needs beyond the binary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StartOptions {
    /// A GGUF to serve. Required by llama.cpp, ignored by the others.
    pub model: Option<PathBuf>,
    pub context_length: Option<usize>,
    /// Layers to offload to the GPU (llama.cpp's `-ngl`).
    pub gpu_layers: Option<usize>,
}

/// Split `http://host:port` into its host and port.
pub fn host_port(url: &str) -> Result<(String, u16)> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or_else(|| {
            Error::invalid(format!(
                "runtime url {url:?} must start with http:// or https://"
            ))
        })?;
    let rest = rest.trim_end_matches('/');
    let (host, port) = rest.rsplit_once(':').ok_or_else(|| {
        Error::invalid(format!(
            "runtime url {url:?} needs a port, e.g. http://127.0.0.1:11434"
        ))
    })?;
    let port = port
        .parse::<u16>()
        .map_err(|_| Error::invalid(format!("runtime url {url:?}: {port:?} is not a port")))?;
    Ok((host.to_string(), port))
}

/// Program arguments and environment for a launch.
pub type Command = (Vec<String>, Vec<(String, String)>);

/// The command that starts a runtime's server at `url`: program arguments
/// and environment, both as the runtime's own documentation describes.
pub fn start_command(kind: RuntimeKind, url: &str, opts: &StartOptions) -> Result<Command> {
    let (host, port) = host_port(url)?;
    match kind {
        RuntimeKind::Ollama => {
            let mut env = Vec::new();
            if url.trim_end_matches('/') != spec(kind).default_url {
                env.push(("OLLAMA_HOST".to_string(), format!("{host}:{port}")));
            }
            Ok((vec!["serve".into()], env))
        }
        RuntimeKind::LlamaCpp => {
            let model = opts.model.as_ref().ok_or_else(|| {
                Error::invalid("llama.cpp needs a model to serve: infy runtime start llamacpp --model <file.gguf>")
            })?;
            let mut args = vec![
                "-m".to_string(),
                model.to_string_lossy().into_owned(),
                "--host".into(),
                host,
                "--port".into(),
                port.to_string(),
            ];
            if let Some(c) = opts.context_length {
                args.push("-c".into());
                args.push(c.to_string());
            }
            if let Some(n) = opts.gpu_layers {
                args.push("-ngl".into());
                args.push(n.to_string());
            }
            Ok((args, Vec::new()))
        }
        RuntimeKind::LmStudio => Ok((
            vec![
                "server".into(),
                "start".into(),
                "--port".into(),
                port.to_string(),
            ],
            Vec::new(),
        )),
    }
}

/// The command a runtime uses to fetch a model into its own store, if it
/// has one. llama.cpp has no registry: its models come from infy's hub.
pub fn pull_command(kind: RuntimeKind, model: &str) -> Result<Vec<String>> {
    match kind {
        RuntimeKind::Ollama => Ok(vec!["pull".into(), model.into()]),
        RuntimeKind::LmStudio => Ok(vec!["get".into(), model.into(), "--yes".into()]),
        RuntimeKind::LlamaCpp => Err(Error::invalid(format!(
            "llama.cpp has no model registry; download {model:?} with `infy pull hf:owner/repo` and start with --model"
        ))),
    }
}

/// How a runtime not started by infy can be stopped, for the refusal
/// message. Killing the user's own service would be worse than refusing.
pub fn foreign_stop_hint(kind: RuntimeKind, os: Os) -> String {
    match (kind, os) {
        (RuntimeKind::Ollama, Os::Linux) => {
            "stop it with `systemctl stop ollama`, or `pkill ollama` if it was started by hand"
                .into()
        }
        (RuntimeKind::Ollama, _) => {
            "quit the Ollama app from the menu bar, or `pkill ollama`".into()
        }
        (RuntimeKind::LlamaCpp, _) => {
            "find it with `pgrep -a llama-server` and stop that process".into()
        }
        (RuntimeKind::LmStudio, _) => {
            "run `lms server stop`, or stop it from the LM Studio app".into()
        }
    }
}

/// A model a running runtime serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedModel {
    pub name: String,
    pub size_bytes: Option<u64>,
    pub detail: Option<String>,
}

#[derive(Deserialize)]
struct OllamaTags {
    #[serde(default)]
    models: Vec<OllamaModel>,
}

#[derive(Deserialize)]
struct OllamaModel {
    name: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    details: Option<OllamaDetails>,
}

#[derive(Deserialize)]
struct OllamaDetails {
    #[serde(default)]
    parameter_size: Option<String>,
    #[serde(default)]
    quantization_level: Option<String>,
    #[serde(default)]
    family: Option<String>,
}

/// Parse a runtime's model listing: ollama's `/api/tags`, or the OpenAI
/// `/v1/models` everyone else serves.
pub fn parse_models(kind: RuntimeKind, body: &str) -> Result<Vec<ServedModel>> {
    match kind {
        RuntimeKind::Ollama => {
            let tags: OllamaTags = serde_json::from_str(body)
                .map_err(|e| Error::wrap("ollama's /api/tags answer did not parse", e))?;
            Ok(tags
                .models
                .into_iter()
                .map(|m| {
                    let detail = m.details.map(|d| {
                        [d.family, d.parameter_size, d.quantization_level]
                            .into_iter()
                            .flatten()
                            .collect::<Vec<_>>()
                            .join(" ")
                    });
                    ServedModel {
                        name: m.name,
                        size_bytes: m.size,
                        detail: detail.filter(|d| !d.is_empty()),
                    }
                })
                .collect())
        }
        RuntimeKind::LlamaCpp | RuntimeKind::LmStudio => {
            let list: infy_wire::ModelList = serde_json::from_str(body).map_err(|e| {
                Error::wrap(
                    format!("{}'s /v1/models answer did not parse", spec(kind).display),
                    e,
                )
            })?;
            Ok(list
                .data
                .into_iter()
                .map(|m| ServedModel {
                    name: m.id,
                    size_bytes: None,
                    detail: if m.owned_by.is_empty() {
                        None
                    } else {
                        Some(m.owned_by)
                    },
                })
                .collect())
        }
    }
}

#[cfg(test)]
#[path = "logic_test.rs"]
mod tests;
