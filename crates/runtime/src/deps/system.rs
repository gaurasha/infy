//! The real machine: `std::process` for processes, `ureq` for HTTP, a file
//! per runtime for pids.

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use infy_kernel::{Cancel, Error, Result};

use crate::deps::{Clock, Http, HttpResponse, Os, Output, PidStore, Processes};
use crate::logic::RuntimeKind;

pub struct SystemProcesses;

impl Processes for SystemProcesses {
    fn os(&self) -> Os {
        if cfg!(target_os = "macos") {
            Os::MacOs
        } else if cfg!(target_os = "windows") {
            Os::Windows
        } else {
            Os::Linux
        }
    }

    fn home(&self) -> Option<PathBuf> {
        std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)
    }

    fn which(&self, name: &str) -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        let candidates: Vec<String> = if cfg!(windows) {
            vec![
                name.to_string(),
                format!("{name}.exe"),
                format!("{name}.cmd"),
            ]
        } else {
            vec![name.to_string()]
        };
        for dir in std::env::split_paths(&path) {
            for c in &candidates {
                let p = dir.join(c);
                if p.is_file() {
                    return Some(p);
                }
            }
        }
        None
    }

    fn exists(&self, path: &Path) -> bool {
        path.is_file()
    }

    fn spawn_detached(
        &self,
        program: &Path,
        args: &[String],
        env: &[(String, String)],
    ) -> Result<u32> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .envs(env.iter().cloned())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            // A new session, so the runtime survives this process exiting
            // and does not receive its terminal's signals.
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let child = cmd.spawn().map_err(|e| {
            Error::wrap(
                format!("starting {} {}", program.display(), args.join(" ")),
                e,
            )
        })?;
        Ok(child.id())
    }

    fn run(
        &self,
        program: &Path,
        args: &[String],
        env: &[(String, String)],
        on_line: &mut dyn FnMut(&str),
    ) -> Result<Output> {
        let mut child = Command::new(program)
            .args(args)
            .envs(env.iter().cloned())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                Error::wrap(
                    format!("running {} {}", program.display(), args.join(" ")),
                    e,
                )
            })?;
        let mut stdout = String::new();
        let mut stderr = String::new();
        // Read stderr on a thread so a chatty command cannot block on a full pipe.
        let err_pipe = child.stderr.take();
        let err_thread = std::thread::spawn(move || {
            let mut s = String::new();
            if let Some(mut p) = err_pipe {
                let _ = p.read_to_string(&mut s);
            }
            s
        });
        if let Some(out) = child.stdout.take() {
            for line in BufReader::new(out)
                .lines()
                .map_while(std::result::Result::ok)
            {
                on_line(&line);
                stdout.push_str(&line);
                stdout.push('\n');
            }
        }
        if let Ok(s) = err_thread.join() {
            for line in s.lines() {
                on_line(line);
            }
            stderr = s;
        }
        let status = child
            .wait()
            .map_err(|e| Error::wrap(format!("waiting for {}", program.display()), e))?;
        Ok(Output {
            status: status.code().unwrap_or(-1),
            stdout,
            stderr,
        })
    }

    fn is_alive(&self, pid: u32) -> bool {
        #[cfg(unix)]
        {
            Path::new(&format!("/proc/{pid}")).exists()
                || Command::new("kill")
                    .args(["-0", &pid.to_string()])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
        }
        #[cfg(not(unix))]
        {
            Command::new("tasklist")
                .args(["/FI", &format!("PID eq {pid}"), "/NH"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()))
                .unwrap_or(false)
        }
    }

    fn kill(&self, pid: u32) -> Result<()> {
        #[cfg(unix)]
        let status = Command::new("kill").arg(pid.to_string()).status();
        #[cfg(not(unix))]
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/F"])
            .status();
        let status = status.map_err(|e| Error::wrap(format!("stopping pid {pid}"), e))?;
        if status.success() {
            Ok(())
        } else {
            Err(Error::unavailable(format!(
                "could not stop pid {pid} (exit {})",
                status.code().unwrap_or(-1)
            )))
        }
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// `ureq`, with HTTP error statuses returned as responses rather than
/// errors so the body can say what went wrong.
pub struct UreqHttp {
    agent: ureq::Agent,
}

impl Default for UreqHttp {
    fn default() -> Self {
        Self::new()
    }
}

impl UreqHttp {
    pub fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(3)))
            .build();
        Self {
            agent: ureq::Agent::new_with_config(config),
        }
    }

    fn map_err(url: &str, e: ureq::Error) -> Error {
        Error::unavailable(format!("{url}: {e}"))
    }
}

impl Http for UreqHttp {
    fn get(&self, url: &str, timeout: Duration) -> Result<HttpResponse> {
        let resp = self
            .agent
            .get(url)
            .config()
            .timeout_global(Some(timeout))
            .build()
            .call()
            .map_err(|e| Self::map_err(url, e))?;
        let status = resp.status().as_u16();
        let body = resp
            .into_body()
            .read_to_string()
            .map_err(|e| Self::map_err(url, e))?;
        Ok(HttpResponse { status, body })
    }

    fn post_json(&self, url: &str, body: &str, timeout: Duration) -> Result<HttpResponse> {
        let resp = self
            .agent
            .post(url)
            .header("content-type", "application/json")
            .config()
            .timeout_global(Some(timeout))
            .build()
            .send(body)
            .map_err(|e| Self::map_err(url, e))?;
        let status = resp.status().as_u16();
        let text = resp
            .into_body()
            .read_to_string()
            .map_err(|e| Self::map_err(url, e))?;
        Ok(HttpResponse { status, body: text })
    }

    fn post_json_stream(
        &self,
        url: &str,
        body: &str,
        cancel: &Cancel,
        on_line: &mut dyn FnMut(&str),
    ) -> Result<u16> {
        let resp = self
            .agent
            .post(url)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .config()
            // A generation can legitimately take minutes; the connect timeout
            // on the agent still bounds an unreachable server.
            .timeout_global(None)
            .build()
            .send(body)
            .map_err(|e| Self::map_err(url, e))?;
        let status = resp.status().as_u16();
        let reader = BufReader::new(resp.into_body().into_reader());
        for line in reader.lines() {
            if cancel.is_cancelled() {
                break;
            }
            let line =
                line.map_err(|e| Error::wrap(format!("reading the stream from {url}"), e))?;
            on_line(&line);
        }
        Ok(status)
    }
}

/// One file per runtime under a directory: `<dir>/ollama.pid`.
pub struct FilePids {
    dir: PathBuf,
}

impl FilePids {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path(&self, kind: RuntimeKind) -> PathBuf {
        self.dir.join(format!("{kind}.pid"))
    }
}

impl PidStore for FilePids {
    fn load(&self, kind: RuntimeKind) -> Result<Option<u32>> {
        match std::fs::read_to_string(self.path(kind)) {
            Ok(s) => Ok(s.trim().parse::<u32>().ok()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::wrap(
                format!("reading {}", self.path(kind).display()),
                e,
            )),
        }
    }

    fn save(&self, kind: RuntimeKind, pid: u32) -> Result<()> {
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| Error::wrap(format!("creating {}", self.dir.display()), e))?;
        std::fs::write(self.path(kind), pid.to_string())
            .map_err(|e| Error::wrap(format!("writing {}", self.path(kind).display()), e))
    }

    fn clear(&self, kind: RuntimeKind) -> Result<()> {
        match std::fs::remove_file(self.path(kind)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::wrap(
                format!("removing {}", self.path(kind).display()),
                e,
            )),
        }
    }
}

/// `Instant` since the first read.
pub struct SystemClock(std::time::Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(std::time::Instant::now())
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
}
