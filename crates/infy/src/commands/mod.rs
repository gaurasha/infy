//! The verbs. Each builds what it needs from the bootstrap values and the
//! adapters, then does one thing.

mod chat;
mod generate;
mod models;
mod pull;
mod runtime;
mod serve;
mod wiring;

use infy_kernel::Result;

use crate::bootstrap::{Bootstrap, Command};

pub fn run(boot: &Bootstrap, command: Command) -> Result<()> {
    match command {
        Command::Chat {
            model,
            system,
            sampling,
        } => chat::run(boot, model, system, sampling),
        Command::Generate {
            prompt,
            model,
            raw,
            sampling,
        } => generate::run(boot, prompt, model, raw, sampling),
        Command::Serve {
            model,
            listen,
            sampling,
        } => serve::run(boot, model, listen, sampling),
        Command::Pull { model } => pull::run(boot, &model),
        Command::Models => models::run(boot),
        Command::Runtime { action } => runtime::run(boot, action),
    }
}
