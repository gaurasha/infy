use super::*;

fn r(s: &str) -> ModelRef {
    ModelRef::new(s).unwrap()
}

#[test]
fn ref_forms() {
    let hf = |repo: &str, revision: Option<&str>, select: GgufSelector| Source::Hf {
        repo: repo.into(),
        revision: revision.map(String::from),
        select,
    };
    assert_eq!(
        parse_ref(&r("hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF")).unwrap(),
        hf(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            None,
            GgufSelector::Default
        )
    );
    assert_eq!(
        parse_ref(&r("Qwen/Qwen2.5-0.5B-Instruct-GGUF")).unwrap(),
        hf(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            None,
            GgufSelector::Default
        )
    );
    assert_eq!(
        parse_ref(&r("hf:o/r:Q8_0")).unwrap(),
        hf("o/r", None, GgufSelector::Quant("Q8_0".into()))
    );
    assert_eq!(
        parse_ref(&r("hf:o/r@main:Q8_0")).unwrap(),
        hf("o/r", Some("main"), GgufSelector::Quant("Q8_0".into()))
    );
    assert_eq!(
        parse_ref(&r("hf:o/r/sub/m.gguf")).unwrap(),
        hf("o/r", None, GgufSelector::File("sub/m.gguf".into()))
    );
    assert_eq!(
        parse_ref(&r("hf:o/r@v2/m.gguf")).unwrap(),
        hf("o/r", Some("v2"), GgufSelector::File("m.gguf".into()))
    );

    assert_eq!(
        parse_ref(&r("./m.gguf")).unwrap(),
        Source::Path("./m.gguf".into())
    );
    assert_eq!(
        parse_ref(&r("/abs/m.gguf")).unwrap(),
        Source::Path("/abs/m.gguf".into())
    );
    assert_eq!(
        parse_ref(&r("~/m.gguf")).unwrap(),
        Source::Path("~/m.gguf".into())
    );
    assert_eq!(
        parse_ref(&r("m.gguf")).unwrap(),
        Source::Path("m.gguf".into())
    );
    assert_eq!(
        parse_ref(&r("../x/m.gguf")).unwrap(),
        Source::Path("../x/m.gguf".into())
    );

    assert_eq!(
        parse_ref(&r("ollama:llama3.2:3b")).unwrap(),
        Source::Runtime {
            runtime: "ollama".into(),
            model: "llama3.2:3b".into()
        }
    );
    assert_eq!(
        parse_ref(&r("llamacpp:default")).unwrap(),
        Source::Runtime {
            runtime: "llamacpp".into(),
            model: "default".into()
        }
    );
    assert_eq!(
        parse_ref(&r("lmstudio:qwen2.5-7b")).unwrap(),
        Source::Runtime {
            runtime: "lmstudio".into(),
            model: "qwen2.5-7b".into()
        }
    );

    assert_eq!(parse_ref(&r("tiny")).unwrap(), Source::Local("tiny".into()));
    assert_eq!(
        parse_ref(&r("o--r/m.gguf")).unwrap(),
        Source::Local("o--r/m.gguf".into()),
        "a repo-shaped name ending in .gguf is a local path"
    );
}

#[test]
fn ref_errors_say_what_is_wrong() {
    for (bad, needle) in [
        ("hf:norepo", "owner/repo"),
        ("hf:o/r:", "quantisation"),
        ("hf:o/r@", "revision"),
        ("hf:o/r/notes.txt", ".gguf"),
        ("ollama:", "model name"),
        ("hf:/r", "owner/repo"),
    ] {
        match parse_ref(&r(bad)) {
            Err(Error::Invalid(m)) => assert!(m.contains(needle), "{bad}: {m}"),
            other => panic!("{bad}: {other:?}"),
        }
    }
}

#[test]
fn local_dir_names_are_flat() {
    assert_eq!(
        local_dir_name("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        "Qwen--Qwen2.5-0.5B-Instruct-GGUF"
    );
}

fn files(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn default_selection_prefers_q4_k_m_then_the_next_usual_then_the_only_file() {
    let repo = files(&[
        "README.md",
        "m-Q8_0.gguf",
        "m-Q4_K_M.gguf",
        "m-Q4_K_S.gguf",
        "mmproj-m-f16.gguf",
    ]);
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Default).unwrap(),
        "m-Q4_K_M.gguf"
    );
    let repo = files(&["m-Q8_0.gguf", "m-Q5_K_M.gguf"]);
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Default).unwrap(),
        "m-Q5_K_M.gguf"
    );
    let repo = files(&["only.gguf", "README.md"]);
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Default).unwrap(),
        "only.gguf"
    );
    let repo = files(&["m-q4_k_m.gguf"]);
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Default).unwrap(),
        "m-q4_k_m.gguf",
        "case-insensitive"
    );
}

#[test]
fn default_selection_errors_list_what_is_there() {
    let repo = files(&["a-IQ2.gguf", "b-IQ3.gguf"]);
    match choose_gguf(&repo, &GgufSelector::Default) {
        Err(Error::Invalid(m)) => {
            assert!(m.contains("a-IQ2.gguf") && m.contains(":<QUANT>"), "{m}")
        }
        other => panic!("{other:?}"),
    }
    let repo = files(&["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]);
    assert!(
        matches!(choose_gguf(&repo, &GgufSelector::Default), Err(Error::Invalid(m)) if m.contains("sharded"))
    );
    let repo = files(&["README.md"]);
    assert!(
        matches!(choose_gguf(&repo, &GgufSelector::Default), Err(Error::NotFound(m)) if m.contains("no .gguf"))
    );
}

#[test]
fn quant_and_file_selection() {
    let repo = files(&[
        "m-Q8_0.gguf",
        "m-Q4_K_M.gguf",
        "m-Q4_K_S.gguf",
        "m-IQ4_XS.gguf",
    ]);
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Quant("q8_0".into())).unwrap(),
        "m-Q8_0.gguf"
    );
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Quant("Q4_K_M".into())).unwrap(),
        "m-Q4_K_M.gguf"
    );
    assert!(
        matches!(choose_gguf(&repo, &GgufSelector::Quant("Q4".into())), Err(Error::Invalid(m)) if m.contains("several"))
    );
    assert!(
        matches!(choose_gguf(&repo, &GgufSelector::Quant("Q2_K".into())), Err(Error::NotFound(m)) if m.contains("available"))
    );
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::File("m-IQ4_XS.gguf".into())).unwrap(),
        "m-IQ4_XS.gguf"
    );
    assert!(matches!(
        choose_gguf(&repo, &GgufSelector::File("nope.gguf".into())),
        Err(Error::NotFound(_))
    ));
    // A tag that is a prefix of another resolves to the exact one.
    let repo = files(&["m-Q4_K.gguf", "m-Q4_K_M.gguf"]);
    assert_eq!(
        choose_gguf(&repo, &GgufSelector::Quant("Q4_K".into())).unwrap(),
        "m-Q4_K.gguf"
    );
}
