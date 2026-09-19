use super::*;
use crate::deps::memory::{MemoryClock, MemoryHttp, MemoryPids, MemoryProcesses};
use crate::deps::Os;
use std::sync::Arc;

/// Shares a fake machine and a fake server with the test while the
/// `Runtimes` under test owns boxed handles to the same objects.
struct Rig {
    procs: Arc<MemoryProcesses>,
    http: Arc<MemoryHttp>,
    pids: Arc<MemoryPids>,
}

struct P(Arc<MemoryProcesses>);
impl Processes for P {
    fn os(&self) -> Os {
        self.0.os()
    }
    fn home(&self) -> Option<PathBuf> {
        self.0.home()
    }
    fn which(&self, n: &str) -> Option<PathBuf> {
        self.0.which(n)
    }
    fn exists(&self, p: &std::path::Path) -> bool {
        self.0.exists(p)
    }
    fn spawn_detached(
        &self,
        p: &std::path::Path,
        a: &[String],
        e: &[(String, String)],
    ) -> Result<u32> {
        self.0.spawn_detached(p, a, e)
    }
    fn run(
        &self,
        p: &std::path::Path,
        a: &[String],
        e: &[(String, String)],
        l: &mut dyn FnMut(&str),
    ) -> Result<crate::deps::Output> {
        self.0.run(p, a, e, l)
    }
    fn is_alive(&self, pid: u32) -> bool {
        self.0.is_alive(pid)
    }
    fn kill(&self, pid: u32) -> Result<()> {
        self.0.kill(pid)
    }
    fn sleep(&self, d: Duration) {
        self.0.sleep(d)
    }
}
struct H(Arc<MemoryHttp>);
impl Http for H {
    fn get(&self, u: &str, t: Duration) -> Result<crate::deps::HttpResponse> {
        self.0.get(u, t)
    }
    fn post_json(&self, u: &str, b: &str, t: Duration) -> Result<crate::deps::HttpResponse> {
        self.0.post_json(u, b, t)
    }
    fn post_json_stream(
        &self,
        u: &str,
        b: &str,
        c: &Cancel,
        l: &mut dyn FnMut(&str),
    ) -> Result<u16> {
        self.0.post_json_stream(u, b, c, l)
    }
}
struct S(Arc<MemoryPids>);
impl PidStore for S {
    fn load(&self, k: RuntimeKind) -> Result<Option<u32>> {
        self.0.load(k)
    }
    fn save(&self, k: RuntimeKind, p: u32) -> Result<()> {
        self.0.save(k, p)
    }
    fn clear(&self, k: RuntimeKind) -> Result<()> {
        self.0.clear(k)
    }
}

fn rig(procs: MemoryProcesses, http: MemoryHttp) -> (Runtimes, Rig) {
    rig_with(procs, http, HashMap::new())
}

fn rig_with(
    procs: MemoryProcesses,
    http: MemoryHttp,
    overrides: HashMap<RuntimeKind, RuntimeConfig>,
) -> (Runtimes, Rig) {
    let r = Rig {
        procs: Arc::new(procs),
        http: Arc::new(http),
        pids: Arc::new(MemoryPids::new()),
    };
    let rt = Runtimes::new(
        Box::new(P(r.procs.clone())),
        Box::new(H(r.http.clone())),
        Box::new(S(r.pids.clone())),
        Box::new(MemoryClock::default()),
        overrides,
    )
    .with_start_timeout(Duration::from_millis(10));
    (rt, r)
}

const TAGS: &str = r#"{"models":[{"name":"llama3.2:3b","size":10,"details":{"family":"llama"}}]}"#;

#[test]
fn status_tells_not_installed_from_installed_from_running() {
    let (rt, _) = rig(MemoryProcesses::new(Os::Linux), MemoryHttp::new());
    let s = rt.status(RuntimeKind::Ollama).unwrap();
    assert_eq!(s.state, State::NotInstalled);
    assert!(s.next_step.contains("install"));
    assert_eq!(s.url, "http://127.0.0.1:11434");

    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new(),
    );
    let s = rt.status(RuntimeKind::Ollama).unwrap();
    assert_eq!(s.state, State::Installed);
    assert_eq!(s.binary, Some(PathBuf::from("/usr/bin/ollama")));
    assert_eq!(s.next_step, "start it: infy runtime start ollama");

    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new()
            .with_get("http://127.0.0.1:11434/", 200, "Ollama is running")
            .with_get("http://127.0.0.1:11434/api/tags", 200, TAGS),
    );
    let s = rt.status(RuntimeKind::Ollama).unwrap();
    assert_eq!(s.state, State::Running);
    assert_eq!(s.models.len(), 1);
    assert_eq!(s.models[0].name, "llama3.2:3b");
    assert_eq!(s.started_by_infy, None);
    assert_eq!(rt.status_all().unwrap().len(), 3);
}

#[test]
fn binaries_are_found_on_path_in_well_known_places_or_by_override() {
    let (rt, _) = rig(
        MemoryProcesses::new(Os::MacOs)
            .with_file("/Applications/Ollama.app/Contents/Resources/ollama"),
        MemoryHttp::new(),
    );
    assert_eq!(
        rt.binary(RuntimeKind::Ollama),
        Some(PathBuf::from(
            "/Applications/Ollama.app/Contents/Resources/ollama"
        ))
    );

    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux).with_file("/home/me/.lmstudio/bin/lms"),
        MemoryHttp::new(),
    );
    assert_eq!(
        rt.binary(RuntimeKind::LmStudio),
        Some(PathBuf::from("/home/me/.lmstudio/bin/lms")),
        "~ expands to the home"
    );
    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux)
            .with_home(None)
            .with_file("/home/me/.lmstudio/bin/lms"),
        MemoryHttp::new(),
    );
    assert_eq!(
        rt.binary(RuntimeKind::LmStudio),
        None,
        "no home, no ~ paths"
    );

    let mut ov = HashMap::new();
    ov.insert(
        RuntimeKind::LlamaCpp,
        RuntimeConfig {
            binary: Some("/opt/llama/llama-server".into()),
            url: Some("http://127.0.0.1:9000/".into()),
        },
    );
    let (rt, _) = rig_with(
        MemoryProcesses::new(Os::Linux).with_binary("llama-server", "/usr/bin/llama-server"),
        MemoryHttp::new(),
        ov,
    );
    assert_eq!(
        rt.binary(RuntimeKind::LlamaCpp),
        Some(PathBuf::from("/opt/llama/llama-server")),
        "an override beats PATH"
    );
    assert_eq!(rt.url(RuntimeKind::LlamaCpp), "http://127.0.0.1:9000");
}

#[test]
fn start_spawns_the_server_waits_for_it_and_remembers_the_pid() {
    let (rt, r) = rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new()
            .with_get("http://127.0.0.1:11434/", 200, "ok")
            .unreachable_for("http://127.0.0.1:11434/", 3),
    );
    let started = rt
        .start(RuntimeKind::Ollama, &StartOptions::default())
        .unwrap();
    let spawned = r.procs.spawned();
    assert_eq!(spawned.len(), 1);
    assert_eq!(spawned[0].program, PathBuf::from("/usr/bin/ollama"));
    assert_eq!(spawned[0].args, vec!["serve"]);
    assert_eq!(
        started,
        Started::Launched {
            pid: spawned[0].pid,
            url: "http://127.0.0.1:11434".into()
        }
    );
    assert_eq!(
        r.pids.load(RuntimeKind::Ollama).unwrap(),
        Some(spawned[0].pid)
    );
    assert!(
        r.procs.slept() >= Duration::from_millis(500),
        "polled at least twice"
    );
    assert_eq!(
        rt.status(RuntimeKind::Ollama).unwrap().started_by_infy,
        Some(spawned[0].pid)
    );

    assert_eq!(
        rt.start(RuntimeKind::Ollama, &StartOptions::default())
            .unwrap(),
        Started::AlreadyRunning {
            url: "http://127.0.0.1:11434".into()
        }
    );
    assert_eq!(r.procs.spawned().len(), 1, "no second launch");
}

#[test]
fn start_llamacpp_needs_a_model_and_lmstudio_uses_lms() {
    let (rt, r) = rig(
        MemoryProcesses::new(Os::Linux)
            .with_binary("llama-server", "/usr/bin/llama-server")
            .with_binary("lms", "/usr/bin/lms"),
        MemoryHttp::new()
            .with_get("http://127.0.0.1:8080/health", 200, "ok")
            .unreachable_for("http://127.0.0.1:8080/health", 2)
            .with_get(
                "http://127.0.0.1:1234/v1/models",
                200,
                r#"{"object":"list","data":[]}"#,
            )
            .unreachable_for("http://127.0.0.1:1234/v1/models", 1),
    );
    assert!(
        matches!(rt.start(RuntimeKind::LlamaCpp, &StartOptions::default()), Err(Error::Invalid(m)) if m.contains("--model"))
    );
    let opts = StartOptions {
        model: Some("/m.gguf".into()),
        ..Default::default()
    };
    rt.start(RuntimeKind::LlamaCpp, &opts).unwrap();
    rt.start(RuntimeKind::LmStudio, &StartOptions::default())
        .unwrap();
    let s = r.procs.spawned();
    assert_eq!(
        s[0].args,
        vec!["-m", "/m.gguf", "--host", "127.0.0.1", "--port", "8080"]
    );
    assert_eq!(s[1].program, PathBuf::from("/usr/bin/lms"));
    assert_eq!(s[1].args, vec!["server", "start", "--port", "1234"]);
}

#[test]
fn start_failures_say_what_happened() {
    let (rt, _) = rig(MemoryProcesses::new(Os::MacOs), MemoryHttp::new());
    match rt.start(RuntimeKind::Ollama, &StartOptions::default()) {
        Err(Error::Unavailable(m)) => assert!(
            m.contains("not installed") && m.contains("brew install ollama"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
    // Never answers: times out, and the message says it is still running.
    let (rt, r) = rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new(),
    );
    match rt.start(RuntimeKind::Ollama, &StartOptions::default()) {
        Err(Error::Unavailable(m)) => {
            assert!(m.contains("did not answer") && m.contains("11434"), "{m}")
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        r.procs.spawned().len(),
        1,
        "it was launched and left running"
    );
}

#[test]
fn start_reports_a_server_that_exits_immediately() {
    struct Dying(Arc<MemoryProcesses>);
    impl Processes for Dying {
        fn os(&self) -> Os {
            self.0.os()
        }
        fn home(&self) -> Option<PathBuf> {
            self.0.home()
        }
        fn which(&self, n: &str) -> Option<PathBuf> {
            self.0.which(n)
        }
        fn exists(&self, p: &std::path::Path) -> bool {
            self.0.exists(p)
        }
        fn spawn_detached(
            &self,
            p: &std::path::Path,
            a: &[String],
            e: &[(String, String)],
        ) -> Result<u32> {
            let pid = self.0.spawn_detached(p, a, e)?;
            self.0.kill(pid)?;
            Ok(pid)
        }
        fn run(
            &self,
            p: &std::path::Path,
            a: &[String],
            e: &[(String, String)],
            l: &mut dyn FnMut(&str),
        ) -> Result<crate::deps::Output> {
            self.0.run(p, a, e, l)
        }
        fn is_alive(&self, pid: u32) -> bool {
            self.0.is_alive(pid)
        }
        fn kill(&self, pid: u32) -> Result<()> {
            self.0.kill(pid)
        }
        fn sleep(&self, d: Duration) {
            self.0.sleep(d)
        }
    }
    let procs = Arc::new(MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"));
    let pids = Arc::new(MemoryPids::new());
    let rt = Runtimes::new(
        Box::new(Dying(procs.clone())),
        Box::new(H(Arc::new(MemoryHttp::new()))),
        Box::new(S(pids.clone())),
        Box::new(MemoryClock::default()),
        HashMap::new(),
    );
    match rt.start(RuntimeKind::Ollama, &StartOptions::default()) {
        Err(Error::Unavailable(m)) => assert!(
            m.contains("exited right after starting") && m.contains("/usr/bin/ollama serve"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
    assert_eq!(pids.load(RuntimeKind::Ollama).unwrap(), None);
}

#[test]
fn stop_kills_only_what_infy_started() {
    let (rt, r) = rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new()
            .with_get("http://127.0.0.1:11434/", 200, "ok")
            .unreachable_for("http://127.0.0.1:11434/", 1),
    );
    let Started::Launched { pid, .. } = rt
        .start(RuntimeKind::Ollama, &StartOptions::default())
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(rt.stop(RuntimeKind::Ollama).unwrap(), Stopped::Killed(pid));
    assert_eq!(r.procs.killed(), vec![pid]);
    assert_eq!(r.pids.load(RuntimeKind::Ollama).unwrap(), None);

    // Still reachable (the fake server keeps answering) but no longer ours:
    // refuse, and name the right command for this OS.
    match rt.stop(RuntimeKind::Ollama) {
        Err(Error::Invalid(m)) => assert!(
            m.contains("not started by infy") && m.contains("systemctl stop ollama"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }

    let (rt, _) = rig(MemoryProcesses::new(Os::Linux), MemoryHttp::new());
    assert_eq!(rt.stop(RuntimeKind::Ollama).unwrap(), Stopped::NotRunning);

    // A stale pid file (process gone) is cleaned up rather than trusted.
    let (rt, r) = rig(MemoryProcesses::new(Os::Linux), MemoryHttp::new());
    r.pids.save(RuntimeKind::LlamaCpp, 999).unwrap();
    assert_eq!(rt.stop(RuntimeKind::LlamaCpp).unwrap(), Stopped::NotRunning);
    assert_eq!(r.pids.load(RuntimeKind::LlamaCpp).unwrap(), None);
    assert!(r.procs.killed().is_empty());
}

#[test]
fn stop_lmstudio_uses_its_own_verb() {
    let (rt, r) = rig(
        MemoryProcesses::new(Os::MacOs)
            .with_binary("lms", "/usr/bin/lms")
            .with_run("lms server stop", 0, &["Stopped the server"]),
        MemoryHttp::new().with_get(
            "http://127.0.0.1:1234/v1/models",
            200,
            "{\"object\":\"list\",\"data\":[]}",
        ),
    );
    assert_eq!(rt.stop(RuntimeKind::LmStudio).unwrap(), Stopped::NotRunning);
    assert_eq!(r.procs.ran(), vec!["lms server stop"]);
}

#[test]
fn install_shows_or_runs_the_command() {
    let (rt, r) = rig(
        MemoryProcesses::new(Os::Linux)
            .with_binary("sh", "/bin/sh")
            .with_run(
                "sh -c curl -fsSL https://ollama.com/install.sh | sh",
                0,
                &[">>> Installing ollama"],
            ),
        MemoryHttp::new(),
    );
    let mut lines = Vec::new();
    assert_eq!(
        rt.install(RuntimeKind::Ollama, false, &mut |l| lines
            .push(l.to_string()))
            .unwrap(),
        Installed::Command("curl -fsSL https://ollama.com/install.sh | sh".into())
    );
    assert!(r.procs.ran().is_empty());
    assert_eq!(
        rt.install(RuntimeKind::Ollama, true, &mut |l| lines
            .push(l.to_string()))
            .unwrap(),
        Installed::Ran
    );
    assert_eq!(lines, vec![">>> Installing ollama"]);
    assert_eq!(
        rt.install(RuntimeKind::LmStudio, true, &mut |_| {})
            .unwrap(),
        Installed::Manual("https://lmstudio.ai/download".into())
    );

    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux)
            .with_binary("sh", "/bin/sh")
            .with_run(
                "sh -c curl -fsSL https://ollama.com/install.sh | sh",
                1,
                &[],
            ),
        MemoryHttp::new(),
    );
    assert!(
        matches!(rt.install(RuntimeKind::Ollama, true, &mut |_| {}), Err(Error::Unavailable(m)) if m.contains("exit 1"))
    );
}

#[test]
fn models_need_a_running_server() {
    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new(),
    );
    match rt.models(RuntimeKind::Ollama) {
        Err(Error::Unavailable(m)) => assert!(
            m.contains("not answering") && m.contains("infy runtime start ollama"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
    let (rt, _) = rig(
        MemoryProcesses::new(Os::Linux),
        MemoryHttp::new()
            .with_get("http://127.0.0.1:8080/health", 200, "ok")
            .with_get("http://127.0.0.1:8080/v1/models", 500, "boom"),
    );
    assert!(
        matches!(rt.models(RuntimeKind::LlamaCpp), Err(Error::Unavailable(m)) if m.contains("500") && m.contains("boom"))
    );
}

#[test]
fn pull_runs_the_runtimes_own_command() {
    let (rt, r) = rig(
        MemoryProcesses::new(Os::Linux)
            .with_binary("ollama", "/usr/bin/ollama")
            .with_run(
                "ollama pull llama3.2:3b",
                0,
                &["pulling manifest", "success"],
            ),
        MemoryHttp::new(),
    );
    let mut lines = Vec::new();
    rt.pull(RuntimeKind::Ollama, "llama3.2:3b", &mut |l| {
        lines.push(l.to_string())
    })
    .unwrap();
    assert_eq!(lines, vec!["pulling manifest", "success"]);
    assert_eq!(r.procs.ran(), vec!["ollama pull llama3.2:3b"]);
    assert!(
        matches!(rt.pull(RuntimeKind::LlamaCpp, "x", &mut |_| {}), Err(Error::Invalid(m)) if m.contains("infy pull hf:"))
    );
    let (rt, _) = rig(MemoryProcesses::new(Os::Linux), MemoryHttp::new());
    assert!(
        matches!(rt.pull(RuntimeKind::Ollama, "x", &mut |_| {}), Err(Error::Unavailable(m)) if m.contains("not installed"))
    );
}

const SSE: &[&str] = &[
    r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
    "",
    r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}"#,
    r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"llama3.2:3b","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}"#,
    r#"data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"llama3.2:3b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":2,"total_tokens":14}}"#,
    "data: [DONE]",
];

fn running_ollama(stream: &[&str], status: u16) -> (Runtimes, Rig) {
    rig(
        MemoryProcesses::new(Os::Linux).with_binary("ollama", "/usr/bin/ollama"),
        MemoryHttp::new()
            .with_get("http://127.0.0.1:11434/", 200, "ok")
            .with_post_stream("http://127.0.0.1:11434/v1/chat/completions", status, stream)
            .with_post_stream("http://127.0.0.1:11434/v1/completions", status, stream),
    )
}

#[test]
fn chat_streams_deltas_and_reports_what_answered() {
    let (rt, r) = running_ollama(SSE, 200);
    let mut seen = String::new();
    let params = SamplingParams {
        max_tokens: 64,
        seed: Some(3),
        stop: vec!["END".into()],
        ..SamplingParams::default()
    };
    let c = rt
        .chat(
            RuntimeKind::Ollama,
            &Cancel::new(),
            "llama3.2",
            &[Message::user("hi")],
            &params,
            &mut |s| seen.push_str(s),
        )
        .unwrap();
    assert_eq!(c.text, "Hello");
    assert_eq!(seen, "Hello");
    assert_eq!(
        c.model, "llama3.2:3b",
        "the resolved tag, not the requested one"
    );
    assert_eq!(c.finish, FinishReason::Stop);
    assert_eq!(
        c.usage,
        Usage {
            prompt_tokens: 12,
            completion_tokens: 2,
            cached_tokens: 0
        }
    );
    assert!(c.timing.total > Duration::ZERO && c.timing.time_to_first_token > Duration::ZERO);

    let (_, body) = &r.http.requests()[1];
    let sent: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(sent["model"], "llama3.2");
    assert_eq!(sent["stream"], true);
    assert_eq!(sent["stream_options"]["include_usage"], true);
    assert_eq!(sent["max_tokens"], 64);
    assert_eq!(sent["seed"], 3);
    assert_eq!(sent["stop"][0], "END");
    assert_eq!(sent["messages"][0]["content"], "hi");
    assert_eq!(c.prompt, *body, "the exact request is the prompt as sent");
}

#[test]
fn chat_errors_are_specific_and_cancellation_stops_the_stream() {
    let (rt, _) = running_ollama(
        &[r#"{"error":{"message":"model 'nope' not found","type":"api_error"}}"#],
        404,
    );
    match rt.chat(
        RuntimeKind::Ollama,
        &Cancel::new(),
        "nope",
        &[Message::user("hi")],
        &SamplingParams::default(),
        &mut |_| {},
    ) {
        Err(Error::NotFound(m)) => assert!(
            m.contains("model 'nope' not found") && m.contains("infy pull ollama:nope"),
            "{m}"
        ),
        other => panic!("{other:?}"),
    }
    let (rt, _) = running_ollama(&["overloaded"], 503);
    assert!(
        matches!(rt.chat(RuntimeKind::Ollama, &Cancel::new(), "m", &[Message::user("hi")], &SamplingParams::default(), &mut |_| {}), Err(Error::Unavailable(m)) if m.contains("503") && m.contains("overloaded"))
    );

    let (rt, _) = running_ollama(SSE, 200);
    let cancel = Cancel::new();
    let inner = cancel.clone();
    let c = rt
        .chat(
            RuntimeKind::Ollama,
            &cancel,
            "m",
            &[Message::user("hi")],
            &SamplingParams::default(),
            &mut |_| inner.cancel(),
        )
        .unwrap();
    assert_eq!(c.text, "Hel", "stopped after the first delta");
    assert_eq!(c.finish, FinishReason::Cancelled);

    let (rt, _) = rig(MemoryProcesses::new(Os::Linux), MemoryHttp::new());
    assert!(
        matches!(rt.chat(RuntimeKind::LmStudio, &Cancel::new(), "m", &[], &SamplingParams::default(), &mut |_| {}), Err(Error::Unavailable(m)) if m.contains("LM Studio"))
    );
    assert!(matches!(
        rt.chat(
            RuntimeKind::Ollama,
            &Cancel::new(),
            "m",
            &[],
            &SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            &mut |_| {}
        ),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn complete_uses_the_completions_endpoint() {
    let stream = &[
        r#"data: {"id":"x","object":"text_completion","created":1,"model":"m:latest","choices":[{"text":"foo","index":0,"finish_reason":null}]}"#,
        r#"data: {"id":"x","object":"text_completion","created":1,"model":"m:latest","choices":[{"text":"","index":0,"finish_reason":"length"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
        "data: [DONE]",
    ];
    let (rt, r) = running_ollama(stream, 200);
    let c = rt
        .complete(
            RuntimeKind::Ollama,
            &Cancel::new(),
            "m",
            "Once",
            &SamplingParams::greedy(),
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(c.text, "foo");
    assert_eq!(c.finish, FinishReason::Length);
    assert_eq!(c.model, "m:latest");
    let (url, body) = &r.http.requests()[1];
    assert_eq!(url, "POST http://127.0.0.1:11434/v1/completions");
    assert!(body.contains("\"prompt\":\"Once\""));
    assert!(!body.contains("top_k"), "greedy sends no top_k");
}
