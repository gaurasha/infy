use super::*;

#[test]
fn overrides_parse_name_and_value() {
    assert_eq!(
        parse_override("ollama=/opt/ollama").unwrap(),
        (RuntimeKind::Ollama, "/opt/ollama".into())
    );
    assert_eq!(
        parse_override("llama.cpp=http://127.0.0.1:9000").unwrap(),
        (RuntimeKind::LlamaCpp, "http://127.0.0.1:9000".into())
    );
    assert!(matches!(parse_override("ollama"), Err(Error::Invalid(m)) if m.contains("name=value")));
    assert!(matches!(parse_override("=x"), Err(Error::Invalid(_))));
    assert!(matches!(parse_override("vllm=x"), Err(Error::Invalid(m)) if m.contains("vllm")));
}

#[test]
fn sampling_flags_validate() {
    let ok = SamplingArgs {
        max_tokens: 3,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        stop: vec!["x".into()],
        seed: Some(1),
    };
    let p = ok.to_params().unwrap();
    assert_eq!(p.max_tokens, 3);
    assert_eq!(p.stop, vec!["x".to_string()]);
    let bad = SamplingArgs { top_p: 2.0, ..ok };
    assert!(matches!(bad.to_params(), Err(Error::Invalid(m)) if m.contains("top_p")));
}

#[test]
fn the_cli_parses_the_documented_forms() {
    let cli = Cli::try_parse_from([
        "infy", "--device", "cpu", "chat", "--model", "hf:o/r", "--system", "be brief", "--seed",
        "3",
    ])
    .unwrap();
    assert_eq!(cli.device, "cpu");
    match cli.command {
        Command::Chat {
            model,
            system,
            sampling,
        } => {
            assert_eq!(model.model.as_deref(), Some("hf:o/r"));
            assert_eq!(system.as_deref(), Some("be brief"));
            assert_eq!(sampling.seed, Some(3));
        }
        other => panic!("{other:?}"),
    }
    let cli = Cli::try_parse_from([
        "infy", "runtime", "start", "llamacpp", "--model", "m.gguf", "--ctx", "4096",
    ])
    .unwrap();
    assert!(
        matches!(cli.command, Command::Runtime { action: RuntimeAction::Start { ref runtime, .. } } if runtime == "llamacpp")
    );
    let cli = Cli::try_parse_from([
        "infy",
        "--runtime-bin",
        "ollama=/x",
        "--runtime-url",
        "ollama=http://h:1",
        "models",
    ])
    .unwrap();
    assert_eq!(cli.runtime_bins, vec!["ollama=/x"]);
    assert_eq!(cli.runtime_urls, vec!["ollama=http://h:1"]);
    assert!(Cli::try_parse_from(["infy"]).is_err(), "a verb is required");
}
