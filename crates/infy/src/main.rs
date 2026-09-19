//! The composition root and the CLI.
//!
//! This is the ONLY crate permitted to depend on every other one, and
//! wiring them together is its entire job. Every domain declares what it
//! needs as a trait and receives a concrete implementation here, which is
//! what lets the domains stay unaware of each other.
//!
//! `bootstrap.rs` is the only place that reads flags or the environment.
//! `adapters.rs` is the only place where two domains meet: newtypes that
//! implement one domain's required trait over another domain's type.
//! `commands/` are the verbs. Reading `adapters.rs` top to bottom is the
//! fastest way to understand how infy fits together.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

mod adapters;
mod bootstrap;
mod commands;

use clap::Parser;

use infy_kernel::Error;

fn main() {
    let cli = bootstrap::Cli::parse();
    let boot = match bootstrap::Bootstrap::from_cli(&cli) {
        Ok(b) => b,
        Err(e) => exit_with(&e),
    };
    boot.init_logging();
    if let Err(e) = commands::run(&boot, cli.command) {
        exit_with(&e);
    }
}

/// Print the error the way the failure discipline wants it -- what is wrong
/// and, where the message carries one, what to do -- and exit with a code
/// that says which kind of failure it was.
fn exit_with(e: &Error) -> ! {
    if e.is_cancelled() {
        eprintln!();
        std::process::exit(130);
    }
    eprintln!("infy: {e}");
    let code = match e {
        Error::Invalid(_) => 2,
        Error::NotFound(_) => 3,
        Error::Unavailable(_) => 4,
        Error::Cancelled => 130,
        Error::Internal { .. } => 1,
    };
    std::process::exit(code);
}
