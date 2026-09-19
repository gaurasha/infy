use super::*;

#[test]
fn kinds_parse_from_what_people_type() {
    assert_eq!(RuntimeKind::parse("ollama").unwrap(), RuntimeKind::Ollama);
    assert_eq!(
        RuntimeKind::parse("llama.cpp").unwrap(),
        RuntimeKind::LlamaCpp
    );
    assert_eq!(
        RuntimeKind::parse("LlamaCpp").unwrap(),
        RuntimeKind::LlamaCpp
    );
    assert_eq!(
        RuntimeKind::parse("lm-studio").unwrap(),
        RuntimeKind::LmStudio
    );
    assert!(matches!(RuntimeKind::parse("vllm"), Err(Error::Invalid(m)) if m.contains("vllm")));
    for k in RuntimeKind::ALL {
        assert_eq!(RuntimeKind::parse(k.name()).unwrap(), k);
        assert_eq!(spec(k).kind, k);
        assert_eq!(k.to_string(), k.name());
    }
}

#[test]
fn start_commands_follow_each_runtimes_documentation() {
    let (args, env) = start_command(
        RuntimeKind::Ollama,
        "http://127.0.0.1:11434",
        &StartOptions::default(),
    )
    .unwrap();
    assert_eq!(args, vec!["serve"]);
    assert!(env.is_empty(), "the default host needs no OLLAMA_HOST");
    let (_, env) = start_command(
        RuntimeKind::Ollama,
        "http://0.0.0.0:11500/",
        &StartOptions::default(),
    )
    .unwrap();
    assert_eq!(
        env,
        vec![("OLLAMA_HOST".to_string(), "0.0.0.0:11500".to_string())]
    );

    let opts = StartOptions {
        model: Some("/m/x.gguf".into()),
        context_length: Some(4096),
        gpu_layers: Some(99),
    };
    let (args, _) = start_command(RuntimeKind::LlamaCpp, "http://127.0.0.1:8080", &opts).unwrap();
    assert_eq!(
        args,
        vec![
            "-m",
            "/m/x.gguf",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "-c",
            "4096",
            "-ngl",
            "99"
        ]
    );
    assert!(matches!(
        start_command(RuntimeKind::LlamaCpp, "http://127.0.0.1:8080", &StartOptions::default()),
        Err(Error::Invalid(m)) if m.contains("--model")
    ));

    let (args, _) = start_command(
        RuntimeKind::LmStudio,
        "http://127.0.0.1:1234",
        &StartOptions::default(),
    )
    .unwrap();
    assert_eq!(args, vec!["server", "start", "--port", "1234"]);
}

#[test]
fn urls_are_split_and_validated() {
    assert_eq!(
        host_port("http://127.0.0.1:11434").unwrap(),
        ("127.0.0.1".to_string(), 11434)
    );
    assert_eq!(
        host_port("https://host.local:8443/").unwrap(),
        ("host.local".to_string(), 8443)
    );
    assert!(matches!(host_port("127.0.0.1:1"), Err(Error::Invalid(m)) if m.contains("http://")));
    assert!(matches!(host_port("http://nohost"), Err(Error::Invalid(m)) if m.contains("port")));
    assert!(matches!(host_port("http://h:x"), Err(Error::Invalid(m)) if m.contains("not a port")));
}

#[test]
fn install_hints_per_os() {
    assert_eq!(
        install_command(RuntimeKind::Ollama, Os::Linux),
        Some("curl -fsSL https://ollama.com/install.sh | sh")
    );
    assert_eq!(
        install_command(RuntimeKind::Ollama, Os::MacOs),
        Some("brew install ollama")
    );
    assert_eq!(install_command(RuntimeKind::Ollama, Os::Windows), None);
    assert!(install_hint(RuntimeKind::Ollama, Os::Windows).contains("ollama.com/download"));
    assert!(install_hint(RuntimeKind::LlamaCpp, Os::MacOs).contains("brew install llama.cpp"));
    assert!(install_hint(RuntimeKind::LlamaCpp, Os::Linux)
        .contains("github.com/ggml-org/llama.cpp/releases"));
    assert!(install_hint(RuntimeKind::LmStudio, Os::Linux).contains("lmstudio.ai"));
}

#[test]
fn next_steps_are_concrete() {
    assert!(
        next_step(RuntimeKind::Ollama, State::NotInstalled, Os::Linux, None).contains("install.sh")
    );
    assert_eq!(
        next_step(RuntimeKind::Ollama, State::Installed, Os::Linux, None),
        "start it: infy runtime start ollama"
    );
    assert!(next_step(
        RuntimeKind::LlamaCpp,
        State::Installed,
        Os::Linux,
        Some("m.gguf")
    )
    .contains("--model m.gguf"));
    assert!(
        next_step(RuntimeKind::LmStudio, State::Running, Os::MacOs, None)
            .contains("lmstudio:<name>")
    );
}

#[test]
fn well_known_paths_expand_home() {
    assert_eq!(
        expand_well_known("~/.ollama/bin/ollama", Some(Path::new("/home/me"))),
        Some(PathBuf::from("/home/me/.ollama/bin/ollama"))
    );
    assert_eq!(expand_well_known("~/x", None), None);
    assert_eq!(
        expand_well_known("/usr/local/bin/ollama", None),
        Some(PathBuf::from("/usr/local/bin/ollama"))
    );
}

#[test]
fn pull_commands_and_foreign_stop_hints() {
    assert_eq!(
        pull_command(RuntimeKind::Ollama, "llama3.2:3b").unwrap(),
        vec!["pull", "llama3.2:3b"]
    );
    assert_eq!(
        pull_command(RuntimeKind::LmStudio, "qwen2.5-7b").unwrap(),
        vec!["get", "qwen2.5-7b", "--yes"]
    );
    assert!(
        matches!(pull_command(RuntimeKind::LlamaCpp, "x"), Err(Error::Invalid(m)) if m.contains("infy pull hf:"))
    );
    assert!(foreign_stop_hint(RuntimeKind::Ollama, Os::Linux).contains("systemctl stop ollama"));
    assert!(foreign_stop_hint(RuntimeKind::Ollama, Os::MacOs).contains("menu bar"));
    assert!(foreign_stop_hint(RuntimeKind::LmStudio, Os::Linux).contains("lms server stop"));
}

#[test]
fn parses_ollama_tags_and_openai_model_lists() {
    let tags = r#"{"models":[{"name":"llama3.2:3b","size":2019393189,"details":{"family":"llama","parameter_size":"3.2B","quantization_level":"Q4_K_M"}},{"name":"nomic-embed-text:latest"}]}"#;
    let m = parse_models(RuntimeKind::Ollama, tags).unwrap();
    assert_eq!(
        m[0],
        ServedModel {
            name: "llama3.2:3b".into(),
            size_bytes: Some(2019393189),
            detail: Some("llama 3.2B Q4_K_M".into())
        }
    );
    assert_eq!(
        m[1],
        ServedModel {
            name: "nomic-embed-text:latest".into(),
            size_bytes: None,
            detail: None
        }
    );
    assert!(parse_models(RuntimeKind::Ollama, "{}").unwrap().is_empty());
    assert!(matches!(
        parse_models(RuntimeKind::Ollama, "nope"),
        Err(Error::Internal { .. })
    ));

    let list = r#"{"object":"list","data":[{"id":"qwen2.5-7b-instruct","object":"model","owned_by":"organization_owner"},{"id":"m2","object":"model"}]}"#;
    let m = parse_models(RuntimeKind::LmStudio, list).unwrap();
    assert_eq!(m[0].name, "qwen2.5-7b-instruct");
    assert_eq!(m[0].detail.as_deref(), Some("organization_owner"));
    assert_eq!(m[1].detail, None);
    assert!(matches!(
        parse_models(RuntimeKind::LlamaCpp, "[]"),
        Err(Error::Internal { .. })
    ));
}
