//! Flags and environment, turned into values. The only place in the whole
//! program that reads either.
//!
//! There is deliberately no configuration file. Bootstrap settings -- where
//! data lives, what to listen on, which device, the hub token, where a
//! runtime's binary is -- come from flags and environment. Everything else
//! is data on disk: a model exists because its file is in the models
//! directory. A config file would be a second source of truth to reconcile
//! with that, and reconciling two sources of truth is a permanent bug
//! factory.

use std::collections::HashMap;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use infy_kernel::{Error, Result, SamplingParams};
use infy_runtime::{RuntimeConfig, RuntimeKind};

#[derive(Parser, Debug)]
#[command(name = "infy", version, about = "A local inference engine", long_about = None)]
pub struct Cli {
    /// Where infy keeps everything: models, runtime state.
    #[arg(long, global = true, env = "INFY_DATA_DIR")]
    pub data_dir: Option<PathBuf>,

    /// Where GGUF files live. Defaults to <data-dir>/models.
    #[arg(long, global = true, env = "INFY_MODELS_DIR")]
    pub models_dir: Option<PathBuf>,

    /// Compute device for native models: auto, cpu, cuda, cuda:N, metal.
    #[arg(long, global = true, env = "INFY_DEVICE", default_value = "auto")]
    pub device: String,

    /// Log filter, e.g. info or infy_engine=debug.
    #[arg(long, global = true, env = "INFY_LOG", default_value = "warn")]
    pub log: String,

    /// Where a runtime's binary is, as name=path (ollama=/opt/ollama/bin/ollama). Repeatable.
    #[arg(long = "runtime-bin", global = true, value_name = "NAME=PATH")]
    pub runtime_bins: Vec<String>,

    /// Where a runtime's server listens, as name=url (llamacpp=http://127.0.0.1:8081). Repeatable.
    #[arg(long = "runtime-url", global = true, value_name = "NAME=URL")]
    pub runtime_urls: Vec<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Talk to a model, turn by turn.
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        /// A system prompt for the conversation.
        #[arg(long)]
        system: Option<String>,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Continue a prompt once and print the result.
    Generate {
        /// The prompt. Read from stdin when omitted.
        prompt: Option<String>,
        #[command(flatten)]
        model: ModelArgs,
        /// Send the prompt as is: no chat template, no BOS.
        #[arg(long)]
        raw: bool,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Serve the OpenAI-compatible API.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        /// Address to listen on.
        #[arg(long, env = "INFY_LISTEN", default_value = "127.0.0.1:8321")]
        listen: String,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Download a model: hf:owner/repo[:QUANT|/file.gguf], or ollama:name.
    Pull {
        /// The model ref.
        model: String,
    },
    /// List local models and what each runtime serves.
    Models,
    /// Detect, start, stop and install external runtimes.
    Runtime {
        #[command(subcommand)]
        action: RuntimeAction,
    },
}

#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    /// Which model: hf:owner/repo, a local name or path, or ollama:/llamacpp:/lmstudio:name.
    #[arg(long, short, env = "INFY_MODEL")]
    pub model: Option<String>,
    /// Print the exact prompt as sent, after the chat template.
    #[arg(long)]
    pub show_prompt: bool,
}

#[derive(Args, Debug, Clone)]
pub struct SamplingArgs {
    #[arg(long, default_value_t = 512)]
    pub max_tokens: usize,
    /// 0 is greedy.
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f32,
    /// 0 disables.
    #[arg(long, default_value_t = 40)]
    pub top_k: usize,
    /// 1.0 disables.
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    /// 1.0 disables.
    #[arg(long, default_value_t = 1.1)]
    pub repetition_penalty: f32,
    /// Stop when this text appears. Repeatable.
    #[arg(long = "stop")]
    pub stop: Vec<String>,
    /// Make the run reproducible.
    #[arg(long)]
    pub seed: Option<u64>,
}

impl SamplingArgs {
    pub fn to_params(&self) -> Result<SamplingParams> {
        let p = SamplingParams {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            stop: self.stop.clone(),
            seed: self.seed,
            ..SamplingParams::default()
        };
        p.validate()?;
        Ok(p)
    }
}

#[derive(Subcommand, Debug)]
pub enum RuntimeAction {
    /// What is installed, what is running, what it serves.
    Status {
        /// One runtime, or all when omitted.
        runtime: Option<String>,
    },
    /// Start a runtime's server and wait until it answers.
    Start {
        runtime: String,
        /// The GGUF to serve (llama.cpp needs one).
        #[arg(long)]
        model: Option<PathBuf>,
        /// Context length to serve with.
        #[arg(long)]
        ctx: Option<usize>,
        /// Layers to offload to the GPU.
        #[arg(long)]
        gpu_layers: Option<usize>,
    },
    /// Stop a runtime infy started.
    Stop { runtime: String },
    /// Show how to install a runtime; run it with --yes.
    Install {
        runtime: String,
        /// Run the install command instead of printing it.
        #[arg(long)]
        yes: bool,
    },
    /// The models a running runtime serves.
    Models { runtime: String },
}

/// Everything the commands need, as plain values.
pub struct Bootstrap {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub device: String,
    pub log: String,
    pub hf_token: Option<String>,
    pub runtime_overrides: HashMap<RuntimeKind, RuntimeConfig>,
}

impl Bootstrap {
    pub fn from_cli(cli: &Cli) -> Result<Self> {
        let data_dir = match &cli.data_dir {
            Some(d) => d.clone(),
            None => default_data_dir()?,
        };
        let models_dir = cli
            .models_dir
            .clone()
            .unwrap_or_else(|| data_dir.join("models"));
        let mut runtime_overrides: HashMap<RuntimeKind, RuntimeConfig> = HashMap::new();
        for spec in &cli.runtime_bins {
            let (kind, value) = parse_override(spec)?;
            runtime_overrides.entry(kind).or_default().binary = Some(PathBuf::from(value));
        }
        for spec in &cli.runtime_urls {
            let (kind, value) = parse_override(spec)?;
            runtime_overrides.entry(kind).or_default().url = Some(value);
        }
        // Read once, then remove: infy spawns runtimes on purpose, and a
        // token left in the environment is inherited by every one of them
        // and readable in /proc/<pid>/environ.
        let hf_token = std::env::var("HF_TOKEN")
            .ok()
            .filter(|t| !t.trim().is_empty());
        std::env::remove_var("HF_TOKEN");
        Ok(Self {
            data_dir,
            models_dir,
            device: cli.device.clone(),
            log: cli.log.clone(),
            hf_token,
            runtime_overrides,
        })
    }

    pub fn init_logging(&self) {
        use tracing_subscriber::EnvFilter;
        let filter = EnvFilter::try_new(&self.log).unwrap_or_else(|_| EnvFilter::new("warn"));
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .try_init();
    }

    pub fn run_dir(&self) -> PathBuf {
        self.data_dir.join("run")
    }

    pub fn hub_cache_dir(&self) -> PathBuf {
        self.data_dir.join("hub-cache")
    }
}

/// `<platform data dir>/infy`, with `~/.infy` as the fallback when the
/// platform has no such notion.
fn default_data_dir() -> Result<PathBuf> {
    if let Some(d) = dirs::data_local_dir() {
        return Ok(d.join("infy"));
    }
    dirs::home_dir()
        .map(|h| h.join(".infy"))
        .ok_or_else(|| Error::invalid("cannot determine a data directory; pass --data-dir"))
}

/// `name=value` for a runtime override.
pub fn parse_override(spec: &str) -> Result<(RuntimeKind, String)> {
    let (name, value) = spec
        .split_once('=')
        .filter(|(n, v)| !n.is_empty() && !v.is_empty())
        .ok_or_else(|| {
            Error::invalid(format!(
                "runtime override {spec:?} must be name=value, e.g. ollama=/usr/local/bin/ollama"
            ))
        })?;
    Ok((RuntimeKind::parse(name)?, value.to_string()))
}

#[cfg(test)]
#[path = "bootstrap_test.rs"]
mod tests;
