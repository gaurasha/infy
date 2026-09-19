//! `infy runtime`: status, start, stop, install, models.

use infy_kernel::Result;
use infy_runtime::{RuntimeKind, StartOptions, Started, State, Stopped};

use crate::bootstrap::{Bootstrap, RuntimeAction};
use crate::commands::wiring::{human_bytes, runtimes};

pub fn run(boot: &Bootstrap, action: RuntimeAction) -> Result<()> {
    let rt = runtimes(boot);
    match action {
        RuntimeAction::Status { runtime } => {
            let statuses = match runtime {
                Some(name) => vec![rt.status(RuntimeKind::parse(&name)?)?],
                None => rt.status_all()?,
            };
            for s in statuses {
                let state = match s.state {
                    State::Running => "running",
                    State::Installed => "installed, not running",
                    State::NotInstalled => "not installed",
                };
                println!("{:<10} {state}", s.display);
                if let Some(b) = &s.binary {
                    println!("           binary  {}", b.display());
                }
                println!("           url     {}", s.url);
                if let Some(pid) = s.started_by_infy {
                    println!("           pid     {pid} (started by infy)");
                }
                if s.state == State::Running {
                    println!(
                        "           models  {}",
                        if s.models.is_empty() {
                            "(none)".to_string()
                        } else {
                            s.models.len().to_string()
                        }
                    );
                }
                println!("           next    {}", s.next_step);
            }
        }
        RuntimeAction::Start {
            runtime,
            model,
            ctx,
            gpu_layers,
        } => {
            let kind = RuntimeKind::parse(&runtime)?;
            let opts = StartOptions {
                model,
                context_length: ctx,
                gpu_layers,
            };
            eprintln!("starting {kind}...");
            match rt.start(kind, &opts)? {
                Started::AlreadyRunning { url } => println!("{kind} is already running at {url}"),
                Started::Launched { pid, url } => println!("{kind} is up at {url} (pid {pid})"),
            }
        }
        RuntimeAction::Stop { runtime } => {
            let kind = RuntimeKind::parse(&runtime)?;
            match rt.stop(kind)? {
                Stopped::Killed(pid) => println!("stopped {kind} (pid {pid})"),
                Stopped::NotRunning => println!("{kind} is not running"),
            }
        }
        RuntimeAction::Install { runtime, yes } => {
            let kind = RuntimeKind::parse(&runtime)?;
            match rt.install(kind, yes, &mut |line| eprintln!("{line}"))? {
                infy_runtime::Installed::Ran => println!("{kind} installed"),
                infy_runtime::Installed::Command(cmd) => {
                    println!("to install {kind}, run:\n\n  {cmd}\n\nor let infy run it: infy runtime install {kind} --yes")
                }
                infy_runtime::Installed::Manual(url) => {
                    println!("{kind} has no command-line installer; download it from {url}")
                }
            }
        }
        RuntimeAction::Models { runtime } => {
            let kind = RuntimeKind::parse(&runtime)?;
            for m in rt.models(kind)? {
                let size = m.size_bytes.map(human_bytes).unwrap_or_default();
                println!(
                    "{kind}:{:<50} {:>10}  {}",
                    m.name,
                    size,
                    m.detail.unwrap_or_default()
                );
            }
        }
    }
    Ok(())
}
