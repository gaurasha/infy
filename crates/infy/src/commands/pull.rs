//! `infy pull`: fetch a model, through the hub or through a runtime.

use infy_hub::{parse_ref, Source};
use infy_kernel::{ModelRef, Result};
use infy_runtime::RuntimeKind;

use crate::bootstrap::Bootstrap;
use crate::commands::wiring::{hub, human_bytes, progress_printer, runtimes};

pub fn run(boot: &Bootstrap, model: &str) -> Result<()> {
    let r = ModelRef::new(model)?;
    if let Source::Runtime { runtime, model } = parse_ref(&r)? {
        let kind = RuntimeKind::parse(&runtime)?;
        runtimes(boot).pull(kind, &model, &mut |line| eprintln!("{line}"))?;
        println!("{kind}:{model}");
        return Ok(());
    }
    let local = hub(boot)?.pull(&r, &mut progress_printer())?;
    eprintln!(
        "{} ({})",
        local.path.display(),
        human_bytes(local.size_bytes)
    );
    println!("{}", local.name);
    Ok(())
}
