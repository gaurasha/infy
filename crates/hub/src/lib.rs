//! Where models come from.
//!
//! Portability: high. Depends on kernel and, behind the `hf` feature, on the
//! `hf-hub` client for the real download adapter. No sibling domain.
//!
//! A user names a model as a [`ModelRef`](infy_kernel::ModelRef); this crate
//! turns it into a [`Source`] (a hub repository, a local file, a name in the
//! models directory, or a model served by an external runtime), picks one
//! GGUF out of a repository's listing, downloads it into the models
//! directory, and lists what is there. The native `models` crate then opens
//! the file; the `runtime` crate handles the runtime sources. Neither is
//! named here.
//!
//! The models directory is the catalogue: a model exists because its file
//! is there. There is no index to keep in sync with the filesystem.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod deps;
mod logic;
mod service;

pub use deps::{Downloader, RemoteFile};
pub use logic::{choose_gguf, local_dir_name, parse_ref, GgufSelector, Source};
pub use service::{Hub, LocalModel};
