//! `infy generate`: one completion to stdout, stats to stderr.

use std::io::{Read, Write};

use infy_engine::{ChatRequest, GenerateRequest};
use infy_kernel::{Error, Message, Result};

use crate::bootstrap::{Bootstrap, ModelArgs, SamplingArgs};
use crate::commands::wiring::{interruptible, runtimes, stats_line, target, Target};

pub fn run(
    boot: &Bootstrap,
    prompt: Option<String>,
    model: ModelArgs,
    raw: bool,
    sampling: SamplingArgs,
) -> Result<()> {
    let params = sampling.to_params()?;
    let prompt = match prompt {
        Some(p) => p,
        None => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| Error::wrap("reading the prompt from stdin", e))?;
            s
        }
    };
    if prompt.trim().is_empty() {
        return Err(Error::invalid("the prompt is empty"));
    }
    let target = target(boot, model.model.as_deref())?;
    let cancel = interruptible();
    let mut out = std::io::stdout();
    let mut sink = |s: &str| {
        let _ = out.write_all(s.as_bytes());
        let _ = out.flush();
    };
    let c = match &target {
        Target::Native(engine) if raw => engine.generate(
            &cancel,
            GenerateRequest {
                prompt: prompt.clone(),
                params,
                raw: true,
            },
            &mut sink,
        )?,
        Target::Native(engine) => engine.chat(
            &cancel,
            ChatRequest {
                messages: vec![Message::user(prompt.clone())],
                params,
            },
            &mut sink,
        )?,
        Target::Runtime(kind, name) if raw => {
            runtimes(boot).complete(*kind, &cancel, name, &prompt, &params, &mut sink)?
        }
        Target::Runtime(kind, name) => runtimes(boot).chat(
            *kind,
            &cancel,
            name,
            &[Message::user(prompt.clone())],
            &params,
            &mut sink,
        )?,
    };
    println!();
    if model.show_prompt {
        eprintln!("--- prompt as sent ---\n{}\n--- end ---", c.prompt);
    }
    eprintln!("{}", stats_line(&c));
    Ok(())
}
