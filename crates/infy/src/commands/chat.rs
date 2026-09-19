//! `infy chat`: a REPL. Streams each answer, prints its usage and timing,
//! keeps the conversation so the prompt cache has something to reuse.

use std::io::Write;

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use infy_engine::ChatRequest;
use infy_kernel::{Completion, Error, Message, Result};

use crate::bootstrap::{Bootstrap, ModelArgs, SamplingArgs};
use crate::commands::wiring::{
    interruptible, not_interruptible, runtimes, stats_line, target, Target,
};

pub fn run(
    boot: &Bootstrap,
    model: ModelArgs,
    system: Option<String>,
    sampling: SamplingArgs,
) -> Result<()> {
    let params = sampling.to_params()?;
    let target = target(boot, model.model.as_deref())?;
    let rt = runtimes(boot);
    let mut history: Vec<Message> = system.into_iter().map(Message::system).collect();
    let mut editor = DefaultEditor::new().map_err(|e| Error::wrap("opening the terminal", e))?;

    eprintln!("infy chat -- /reset clears the conversation, /prompt shows the last prompt as sent, /quit leaves");
    let mut last: Option<Completion> = None;
    loop {
        not_interruptible();
        let line = match editor.readline("you> ") {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => return Err(Error::wrap("reading input", e)),
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(&line);
        match line.as_str() {
            "/quit" | "/exit" => break,
            "/reset" => {
                history.retain(|m| m.role == infy_kernel::Role::System);
                if let Target::Native(e) = &target {
                    e.reset_cache();
                }
                eprintln!("(conversation cleared)");
                continue;
            }
            "/prompt" => {
                match &last {
                    Some(c) => eprintln!("--- prompt as sent ---\n{}\n--- end ---", c.prompt),
                    None => eprintln!("(nothing sent yet)"),
                }
                continue;
            }
            _ => {}
        }
        history.push(Message::user(line));
        let cancel = interruptible();
        let mut out = std::io::stdout();
        let mut sink = |s: &str| {
            let _ = out.write_all(s.as_bytes());
            let _ = out.flush();
        };
        let result = match &target {
            Target::Native(engine) => engine.chat(
                &cancel,
                ChatRequest {
                    messages: history.clone(),
                    params: params.clone(),
                },
                &mut sink,
            ),
            Target::Runtime(kind, name) => {
                rt.chat(*kind, &cancel, name, &history, &params, &mut sink)
            }
        };
        println!();
        match result {
            Ok(c) => {
                if model.show_prompt {
                    eprintln!("--- prompt as sent ---\n{}\n--- end ---", c.prompt);
                }
                eprintln!("{}", stats_line(&c));
                history.push(Message::assistant(c.text.clone()));
                last = Some(c);
            }
            Err(e) if e.is_cancelled() => {
                eprintln!("(interrupted)");
                history.pop();
            }
            Err(e) => {
                eprintln!("infy: {e}");
                history.pop();
            }
        }
    }
    Ok(())
}
