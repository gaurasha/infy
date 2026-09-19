//! External inference engines -- ollama, llama.cpp's `llama-server`, LM
//! Studio -- driven as integrations.
//!
//! Portability: high. Depends on kernel, wire (the protocol every one of
//! these runtimes speaks) and, behind the `system` feature, on `ureq` for
//! the real HTTP adapter. Processes are spawned through `std`. No sibling
//! domain.
//!
//! The shape follows from one observation: these runtimes differ by a
//! binary name, a port, a start command and a models endpoint, and agree on
//! everything else -- they all serve the OpenAI chat-completions protocol.
//! So each is a [`RuntimeSpec`] row in `logic.rs`, and the code that
//! detects, starts, stops, installs and talks to a runtime is written once
//! in `service.rs`. **Adding a runtime is adding a spec.**
//!
//! What the domain requires from outside is declared in `deps.rs`: a way to
//! find and run processes, an HTTP client, and somewhere to remember the
//! pid of a process it started. `deps/memory.rs` implements all three so
//! the tests run with no ollama, no network and no shell.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod deps;
mod logic;
mod service;

pub use deps::{Clock, Http, HttpResponse, Os, Output, PidStore, Processes};
pub use logic::{
    install_command, install_hint, parse_models, spec, start_command, Install, RuntimeKind,
    RuntimeSpec, ServedModel, StartOptions, State, SPECS,
};
pub use service::{RuntimeConfig, Runtimes, Started, Status, Stopped};
