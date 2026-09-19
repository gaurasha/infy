//! In-memory implementations of every trait the runtime domain requires: a
//! fake machine (binaries, processes, home directory, OS), a fake HTTP
//! server with canned routes, and a pid store in a map. They ship with the
//! domain so its tests run with no ollama, no network and no shell.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use infy_kernel::{Cancel, Error, Result};

use crate::deps::{Clock, Http, HttpResponse, Os, Output, PidStore, Processes};
use crate::logic::RuntimeKind;

/// A recorded launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spawned {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub pid: u32,
}

/// A fake machine.
pub struct MemoryProcesses {
    os: Os,
    home: Option<PathBuf>,
    on_path: HashMap<String, PathBuf>,
    files: HashSet<PathBuf>,
    /// Canned results for `run`, keyed by `program args...` as one string.
    runs: HashMap<String, (Output, Vec<String>)>,
    spawned: Mutex<Vec<Spawned>>,
    ran: Mutex<Vec<String>>,
    alive: Mutex<HashSet<u32>>,
    killed: Mutex<Vec<u32>>,
    next_pid: Mutex<u32>,
    slept: Mutex<Duration>,
}

impl MemoryProcesses {
    pub fn new(os: Os) -> Self {
        Self {
            os,
            home: Some(PathBuf::from("/home/me")),
            on_path: HashMap::new(),
            files: HashSet::new(),
            runs: HashMap::new(),
            spawned: Mutex::new(Vec::new()),
            ran: Mutex::new(Vec::new()),
            alive: Mutex::new(HashSet::new()),
            killed: Mutex::new(Vec::new()),
            next_pid: Mutex::new(4242),
            slept: Mutex::new(Duration::ZERO),
        }
    }

    /// An executable on `PATH`.
    pub fn with_binary(mut self, name: &str, path: &str) -> Self {
        self.on_path.insert(name.into(), PathBuf::from(path));
        self.files.insert(PathBuf::from(path));
        self
    }

    /// A file that exists but is not on `PATH`.
    pub fn with_file(mut self, path: &str) -> Self {
        self.files.insert(PathBuf::from(path));
        self
    }

    pub fn with_home(mut self, home: Option<&str>) -> Self {
        self.home = home.map(PathBuf::from);
        self
    }

    /// What `run` answers for a command, plus the lines it streams first.
    pub fn with_run(mut self, command: &str, status: i32, lines: &[&str]) -> Self {
        let out = Output {
            status,
            stdout: lines.join("\n"),
            stderr: String::new(),
        };
        self.runs.insert(
            command.into(),
            (out, lines.iter().map(|s| s.to_string()).collect()),
        );
        self
    }

    /// A pid that is alive although infy did not start it.
    pub fn with_alive(self, pid: u32) -> Self {
        if let Ok(mut a) = self.alive.lock() {
            a.insert(pid);
        }
        self
    }

    pub fn spawned(&self) -> Vec<Spawned> {
        self.spawned.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn ran(&self) -> Vec<String> {
        self.ran.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn killed(&self) -> Vec<u32> {
        self.killed.lock().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn slept(&self) -> Duration {
        self.slept.lock().map(|s| *s).unwrap_or_default()
    }
}

fn command_key(program: &Path, args: &[String]) -> String {
    let name = program.file_name().and_then(|f| f.to_str()).unwrap_or("");
    std::iter::once(name.to_string())
        .chain(args.iter().cloned())
        .collect::<Vec<_>>()
        .join(" ")
}

impl Processes for MemoryProcesses {
    fn os(&self) -> Os {
        self.os
    }
    fn home(&self) -> Option<PathBuf> {
        self.home.clone()
    }
    fn which(&self, name: &str) -> Option<PathBuf> {
        self.on_path.get(name).cloned()
    }
    fn exists(&self, path: &Path) -> bool {
        self.files.contains(path)
    }
    fn spawn_detached(
        &self,
        program: &Path,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<u32> {
        if !self.files.contains(program) {
            return Err(Error::not_found(format!(
                "executable {}",
                program.display()
            )));
        }
        let pid = {
            let mut n = self
                .next_pid
                .lock()
                .map_err(|_| Error::internal("poisoned"))?;
            *n += 1;
            *n
        };
        if let Ok(mut a) = self.alive.lock() {
            a.insert(pid);
        }
        if let Ok(mut s) = self.spawned.lock() {
            s.push(Spawned {
                program: program.to_path_buf(),
                args: args.to_vec(),
                env: env.to_vec(),
                pid,
            });
        }
        Ok(pid)
    }
    fn run(
        &self,
        program: &Path,
        args: &[String],
        _env: &[(String, String)],
        on_line: &mut dyn FnMut(&str),
    ) -> Result<Output> {
        let key = command_key(program, args);
        if let Ok(mut r) = self.ran.lock() {
            r.push(key.clone());
        }
        match self.runs.get(&key) {
            Some((out, lines)) => {
                for l in lines {
                    on_line(l);
                }
                Ok(out.clone())
            }
            None => Err(Error::not_found(format!(
                "no canned result for command {key:?}"
            ))),
        }
    }
    fn is_alive(&self, pid: u32) -> bool {
        self.alive.lock().map(|a| a.contains(&pid)).unwrap_or(false)
    }
    fn kill(&self, pid: u32) -> Result<()> {
        if let Ok(mut a) = self.alive.lock() {
            a.remove(&pid);
        }
        if let Ok(mut k) = self.killed.lock() {
            k.push(pid);
        }
        Ok(())
    }
    fn sleep(&self, d: Duration) {
        if let Ok(mut s) = self.slept.lock() {
            *s += d;
        }
    }
}

/// A canned HTTP server: routes keyed by method and URL.
#[derive(Default)]
pub struct MemoryHttp {
    routes: Mutex<HashMap<String, Route>>,
    requests: Mutex<Vec<(String, String)>>,
    /// GETs to a URL fail this many times before the route answers, to
    /// simulate a server that is still starting.
    unreachable_for: Mutex<HashMap<String, usize>>,
}

enum Route {
    Response(HttpResponse),
    Stream(u16, Vec<String>),
}

impl MemoryHttp {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_get(self, url: &str, status: u16, body: &str) -> Self {
        self.route(
            "GET",
            url,
            Route::Response(HttpResponse {
                status,
                body: body.into(),
            }),
        )
    }

    pub fn with_post(self, url: &str, status: u16, body: &str) -> Self {
        self.route(
            "POST",
            url,
            Route::Response(HttpResponse {
                status,
                body: body.into(),
            }),
        )
    }

    /// A streaming POST: the lines of the body, as an SSE server would send.
    pub fn with_post_stream(self, url: &str, status: u16, lines: &[&str]) -> Self {
        self.route(
            "POST",
            url,
            Route::Stream(status, lines.iter().map(|s| s.to_string()).collect()),
        )
    }

    /// Make a URL refuse connections for the first `n` GETs.
    pub fn unreachable_for(self, url: &str, n: usize) -> Self {
        if let Ok(mut u) = self.unreachable_for.lock() {
            u.insert(url.into(), n);
        }
        self
    }

    fn route(self, method: &str, url: &str, route: Route) -> Self {
        if let Ok(mut r) = self.routes.lock() {
            r.insert(format!("{method} {url}"), route);
        }
        self
    }

    /// Every `(method, body)` posted, in order.
    pub fn requests(&self) -> Vec<(String, String)> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }

    fn lookup(&self, method: &str, url: &str, body: &str) -> Result<HttpResponse> {
        if let Ok(mut r) = self.requests.lock() {
            r.push((format!("{method} {url}"), body.to_string()));
        }
        if method == "GET" {
            if let Ok(mut u) = self.unreachable_for.lock() {
                if let Some(n) = u.get_mut(url) {
                    if *n > 0 {
                        *n -= 1;
                        return Err(Error::unavailable(format!("connection refused: {url}")));
                    }
                }
            }
        }
        let routes = self
            .routes
            .lock()
            .map_err(|_| Error::internal("poisoned"))?;
        match routes.get(&format!("{method} {url}")) {
            Some(Route::Response(r)) => Ok(r.clone()),
            Some(Route::Stream(status, lines)) => Ok(HttpResponse {
                status: *status,
                body: lines.join("\n"),
            }),
            None => Err(Error::unavailable(format!("connection refused: {url}"))),
        }
    }
}

impl Http for MemoryHttp {
    fn get(&self, url: &str, _timeout: Duration) -> Result<HttpResponse> {
        self.lookup("GET", url, "")
    }

    fn post_json(&self, url: &str, body: &str, _timeout: Duration) -> Result<HttpResponse> {
        self.lookup("POST", url, body)
    }

    fn post_json_stream(
        &self,
        url: &str,
        body: &str,
        cancel: &Cancel,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<u16> {
        let r = self.lookup("POST", url, body)?;
        for line in r.body.lines() {
            if cancel.is_cancelled() {
                break;
            }
            on_line(line);
        }
        Ok(r.status)
    }
}

#[derive(Default)]
pub struct MemoryPids(Mutex<HashMap<RuntimeKind, u32>>);

impl MemoryPids {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PidStore for MemoryPids {
    fn load(&self, kind: RuntimeKind) -> Result<Option<u32>> {
        Ok(self
            .0
            .lock()
            .map_err(|_| Error::internal("poisoned"))?
            .get(&kind)
            .copied())
    }
    fn save(&self, kind: RuntimeKind, pid: u32) -> Result<()> {
        self.0
            .lock()
            .map_err(|_| Error::internal("poisoned"))?
            .insert(kind, pid);
        Ok(())
    }
    fn clear(&self, kind: RuntimeKind) -> Result<()> {
        self.0
            .lock()
            .map_err(|_| Error::internal("poisoned"))?
            .remove(&kind);
        Ok(())
    }
}

/// Advances a fixed step per read.
pub struct MemoryClock {
    now: Mutex<Duration>,
    step: Duration,
}

impl Default for MemoryClock {
    fn default() -> Self {
        Self {
            now: Mutex::new(Duration::ZERO),
            step: Duration::from_millis(1),
        }
    }
}

impl Clock for MemoryClock {
    fn now(&self) -> Duration {
        let mut n = match self.now.lock() {
            Ok(n) => n,
            Err(p) => p.into_inner(),
        };
        *n += self.step;
        *n
    }
}
