//! `infy models`: what is here, and what each runtime serves.

use infy_kernel::Result;
use infy_runtime::State;

use crate::bootstrap::Bootstrap;
use crate::commands::wiring::{hub, human_bytes, runtimes};

pub fn run(boot: &Bootstrap) -> Result<()> {
    let local = hub(boot)?.list()?;
    println!("local ({})", boot.models_dir.display());
    if local.is_empty() {
        println!("  (none -- `infy pull hf:owner/repo` downloads one)");
    }
    for m in &local {
        println!("  {:<60} {:>10}", m.name, human_bytes(m.size_bytes));
    }
    let rt = runtimes(boot);
    for s in rt.status_all()? {
        let state = match s.state {
            State::Running => "running",
            State::Installed => "installed, not running",
            State::NotInstalled => "not installed",
        };
        println!("\n{} ({state}, {})", s.display, s.url);
        if s.state == State::Running && s.models.is_empty() {
            println!("  (serves nothing yet)");
        }
        for m in &s.models {
            let size = m.size_bytes.map(human_bytes).unwrap_or_default();
            let detail = m.detail.clone().unwrap_or_default();
            println!("  {}:{:<50} {:>10}  {}", s.kind, m.name, size, detail);
        }
        if s.state != State::Running {
            println!("  {}", s.next_step);
        }
    }
    Ok(())
}
