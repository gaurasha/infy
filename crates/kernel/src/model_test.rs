use super::*;

#[test]
fn model_ref_accepts_the_shapes_users_type() {
    for ok in [
        "hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "./m.gguf",
        "ollama:llama3.2:3b",
        "tiny",
    ] {
        assert_eq!(ModelRef::new(ok).unwrap().as_str(), ok);
    }
}

#[test]
fn model_ref_trims_and_rejects_whitespace_inside() {
    assert_eq!(ModelRef::new("  tiny \n").unwrap().as_str(), "tiny");
    for bad in ["", "   ", "a b", "a\tb", "a\u{7}b"] {
        assert!(
            matches!(ModelRef::new(bad), Err(Error::Invalid(_))),
            "{bad:?}"
        );
    }
}

#[test]
fn model_ref_serialises_as_a_bare_string() {
    let r: ModelRef = "x/y".parse().unwrap();
    assert_eq!(serde_json::to_string(&r).unwrap(), "\"x/y\"");
    let back: ModelRef = serde_json::from_str("\"x/y\"").unwrap();
    assert_eq!(back, r);
}

#[test]
fn vocabulary_eos_is_a_set() {
    let v = Vocabulary {
        eos_token_ids: vec![2, 128009],
        ..Default::default()
    };
    assert!(v.is_eos(2));
    assert!(v.is_eos(128009));
    assert!(!v.is_eos(1));
}
