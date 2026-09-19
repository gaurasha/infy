use std::path::{Path, PathBuf};
use std::time::Duration;

use infy_kernel::{Cancel, Result};

use crate::logic::RuntimeKind;

/// The operating system, for install commands and well-known paths. A value
/// rather than `cfg!` so the pure logic can be tested for every OS on one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    Linux,
    MacOs,
    Windows,
}

/// What a finished command produced.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Output {
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Finding, starting, watching and stopping processes on this machine.
pub trait Processes: Send + Sync {
    fn os(&self) -> Os;
    fn home(&self) -> Option<PathBuf>;
    /// Search `PATH` for an executable.
    fn which(&self, name: &str) -> Option<PathBuf>;
    fn exists(&self, path: &Path) -> bool;
    /// Start a long-running process that outlives this call; return its pid.
    fn spawn_detached(
        &self,
        program: &Path,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<u32>;
    /// Run a command to completion, handing each output line to `on_line`
    /// as it arrives.
    fn run(
        &self,
        program: &Path,
        args: &[String],
        env: &[(String, String)],
        on_line: &mut dyn FnMut(&str),
    ) -> Result<Output>;
    fn is_alive(&self, pid: u32) -> bool;
    fn kill(&self, pid: u32) -> Result<()>;
    fn sleep(&self, d: Duration);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// An HTTP client. Connection failures are `Unavailable`; an HTTP error
/// status is a normal response with that status, so the body can say why.
pub trait Http: Send + Sync {
    fn get(&self, url: &str, timeout: Duration) -> Result<HttpResponse>;
    fn post_json(&self, url: &str, body: &str, timeout: Duration) -> Result<HttpResponse>;
    /// POST and hand over the response body line by line as it streams;
    /// stop early once `cancel` is set. Returns the status.
    fn post_json_stream(
        &self,
        url: &str,
        body: &str,
        cancel: &Cancel,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<u16>;
}

/// Where the pid of a runtime infy started is remembered between
/// invocations, so `stop` only ever kills what `start` launched.
pub trait PidStore: Send + Sync {
    fn load(&self, kind: RuntimeKind) -> Result<Option<u32>>;
    fn save(&self, kind: RuntimeKind, pid: u32) -> Result<()>;
    fn clear(&self, kind: RuntimeKind) -> Result<()>;
}

/// A monotonic clock, for timing every call.
pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
}

pub mod memory;
#[cfg(feature = "system")]
pub mod system;
