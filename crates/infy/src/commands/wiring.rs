//! Construction, shared by the verbs: the hub, the runtimes, a loaded
//! native engine, and the small output helpers.

use std::io::Write;
use std::sync::Arc;

use infy_engine::deps::tokenizers::HfTokenizer;
use infy_engine::Engine;
use infy_hub::deps::hfhub::HfDownloader;
use infy_hub::{parse_ref, Hub, Source};
use infy_kernel::{Cancel, Completion, Error, ModelRef, Result};
use infy_models::deps::gguf::GgufFile;
use infy_models::{vocabulary_from_metadata, LoadOptions, ModelConfig, WeightSource};
use infy_runtime::deps::system::{FilePids, SystemProcesses, UreqHttp};
use infy_runtime::{RuntimeKind, Runtimes};

use crate::adapters::{CandleModel, SystemClock};
use crate::bootstrap::Bootstrap;

pub fn hub(boot: &Bootstrap) -> Result<Hub> {
    let downloader = HfDownloader::new(boot.hf_token.clone(), Some(boot.hub_cache_dir()))?;
    Ok(Hub::new(Box::new(downloader), boot.models_dir.clone()))
}

pub fn runtimes(boot: &Bootstrap) -> Runtimes {
    Runtimes::new(
        Box::new(SystemProcesses),
        Box::new(UreqHttp::new()),
        Box::new(FilePids::new(boot.run_dir())),
        Box::new(SystemClock::default()),
        boot.runtime_overrides.clone(),
    )
}

/// What a `--model` names, once resolved: a native engine, or a runtime and
/// a model name on it.
pub enum Target {
    Native(Arc<Engine>),
    Runtime(RuntimeKind, String),
}

/// Resolve `--model`. With no ref, the only local model is used when there
/// is exactly one; otherwise the error says what to do.
pub fn target(boot: &Bootstrap, model: Option<&str>) -> Result<Target> {
    let hub = hub(boot)?;
    let r = match model {
        Some(m) => ModelRef::new(m)?,
        None => {
            let local = hub.list()?;
            match local.as_slice() {
                [one] => ModelRef::new(one.name.clone())?,
                [] => {
                    return Err(Error::not_found(format!(
                        "no model: nothing in {} and no --model given. Pull one with `infy pull hf:owner/repo`, or use a runtime's model with --model ollama:<name>",
                        boot.models_dir.display()
                    )))
                }
                many => {
                    return Err(Error::invalid(format!(
                        "{} local models; choose one with --model: {}",
                        many.len(),
                        many.iter().map(|m| m.name.as_str()).collect::<Vec<_>>().join(", ")
                    )))
                }
            }
        }
    };
    if let Source::Runtime { runtime, model } = parse_ref(&r)? {
        return Ok(Target::Runtime(RuntimeKind::parse(&runtime)?, model));
    }
    let local = hub.resolve(&r, &mut progress_printer())?;
    Ok(Target::Native(Arc::new(load_engine(
        boot,
        &local.path,
        &local.name,
    )?)))
}

/// Open a GGUF, build the model on the device, rebuild its tokenizer.
pub fn load_engine(boot: &Bootstrap, path: &std::path::Path, id: &str) -> Result<Engine> {
    let file = GgufFile::open(path)?;
    let config = ModelConfig::from_metadata(file.metadata())?;
    let vocab = vocabulary_from_metadata(file.metadata())?;
    let device = infy_models::device(&boot.device)?;
    eprintln!(
        "loading {} ({}: {} layers, {} heads, context {}) on {}",
        id,
        config.architecture.gguf_name(),
        config.num_layers,
        config.num_heads,
        config.context_length,
        if device.is_cpu() { "cpu" } else { &boot.device }
    );
    let started = std::time::Instant::now();
    let llama = infy_models::load(
        &file,
        &LoadOptions {
            device,
            dtype: infy_models::DType::F32,
        },
    )?;
    let tokenizer = HfTokenizer::from_vocabulary(vocab)?;
    eprintln!("loaded in {:.1}s", started.elapsed().as_secs_f64());
    Ok(Engine::new(
        Box::new(CandleModel::new(llama, id.to_string())),
        Box::new(tokenizer),
        Box::new(SystemClock::default()),
    ))
}

/// A download progress line that rewrites itself.
pub fn progress_printer() -> impl FnMut(&str, u64, u64) {
    let mut last_pct = u64::MAX;
    move |file: &str, done: u64, total: u64| {
        let pct = (done * 100).checked_div(total).unwrap_or(0);
        if pct != last_pct {
            last_pct = pct;
            eprint!(
                "\r{file}: {} / {} ({pct}%)   ",
                human_bytes(done),
                human_bytes(total)
            );
            if done >= total && total > 0 {
                eprintln!();
            }
            let _ = std::io::stderr().flush();
        }
    }
}

pub fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1000.0 && i < UNITS.len() - 1 {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}

/// The per-call numbers, one line: what every optimisation is judged by.
pub fn stats_line(c: &Completion) -> String {
    let t = &c.timing;
    format!(
        "[{} · in {} · out {} · cached {} · ttft {:.2}s · {:.1} tok/s · {:?}]",
        c.model,
        c.usage.prompt_tokens,
        c.usage.completion_tokens,
        c.usage.cached_tokens,
        t.time_to_first_token.as_secs_f64(),
        t.decode_tokens_per_second(c.usage.completion_tokens),
        c.finish
    )
}

/// A cancel flag that Ctrl-C sets, so an interrupted generation stops and
/// the program keeps running. Installed once; each generation swaps in its
/// own flag.
pub fn interruptible() -> Cancel {
    use std::sync::{Mutex, OnceLock};
    static CURRENT: OnceLock<Mutex<Option<Cancel>>> = OnceLock::new();
    let slot = CURRENT.get_or_init(|| {
        let slot: Mutex<Option<Cancel>> = Mutex::new(None);
        let _ = ctrlc::set_handler(|| {
            if let Some(m) = CURRENT.get() {
                if let Ok(guard) = m.lock() {
                    match guard.as_ref() {
                        Some(c) if !c.is_cancelled() => c.cancel(),
                        // Nothing running, or already asked once: quit.
                        _ => std::process::exit(130),
                    }
                }
            }
        });
        slot
    });
    let cancel = Cancel::new();
    if let Ok(mut g) = slot.lock() {
        *g = Some(cancel.clone());
    }
    cancel
}

/// Forget the current generation's flag, so the next Ctrl-C quits.
pub fn not_interruptible() {
    // Same slot as above; reached through a fresh flag that is immediately
    // marked cancelled, which the handler treats as "quit".
    let c = interruptible();
    c.cancel();
}
