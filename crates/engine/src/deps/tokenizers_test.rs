use super::*;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;

/// A byte-level BPE vocabulary built by hand: the 256 byte symbols, a few
/// merges, and two control tokens.
fn gpt2_vocab(pre: Option<&str>) -> Vocabulary {
    let mut tokens: Vec<String> = ByteLevel::alphabet()
        .into_iter()
        .map(String::from)
        .collect();
    tokens.sort();
    let mut types = vec![1; tokens.len()];
    let merges = vec![
        "h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d",
    ];
    for m in &merges {
        tokens.push(m.replace(' ', ""));
        types.push(1);
    }
    for special in ["<|im_start|>", "<|im_end|>"] {
        tokens.push(special.into());
        types.push(TYPE_CONTROL);
    }
    let eos = tokens.iter().position(|t| t == "<|im_end|>").unwrap() as u32;
    Vocabulary {
        model: "gpt2".into(),
        pre: pre.map(String::from),
        tokens,
        token_types: types,
        merges: merges.iter().map(|m| m.to_string()).collect(),
        eos_token_ids: vec![eos],
        add_bos_token: Some(false),
        chat_template: Some("{{ messages }}".into()),
        ..Default::default()
    }
}

#[test]
fn byte_level_bpe_merges_and_round_trips() {
    let tk = HfTokenizer::from_vocabulary(gpt2_vocab(None)).unwrap();
    let ids = tk.encode("hello world").unwrap();
    let hello = tk.vocab.tokens.iter().position(|t| t == "hello").unwrap() as u32;
    let world = tk.vocab.tokens.iter().position(|t| t == "Ġworld").unwrap() as u32;
    assert_eq!(
        ids,
        vec![hello, world],
        "merges apply; the space is folded into Ġworld"
    );
    assert_eq!(tk.decode(&ids).unwrap(), "hello world");
    assert_eq!(
        tk.decode(&tk.encode("zebra!! é").unwrap()).unwrap(),
        "zebra!! é",
        "unknown words fall back to bytes"
    );
    assert!(!tk.add_bos());
    assert_eq!(tk.chat_template().as_deref(), Some("{{ messages }}"));
}

#[test]
fn control_tokens_are_single_tokens_and_are_skipped_on_decode() {
    let tk = HfTokenizer::from_vocabulary(gpt2_vocab(Some("llama-bpe"))).unwrap();
    let ids = tk.encode("<|im_start|>hello<|im_end|>").unwrap();
    let start = tk
        .vocab
        .tokens
        .iter()
        .position(|t| t == "<|im_start|>")
        .unwrap() as u32;
    assert_eq!(ids.first(), Some(&start));
    assert_eq!(ids.last(), Some(&tk.eos_tokens()[0]));
    assert_eq!(ids.len(), 3);
    assert_eq!(tk.decode(&ids).unwrap(), "hello");
    assert_eq!(
        tk.special_token_text(start).as_deref(),
        Some("<|im_start|>")
    );
}

#[test]
fn incomplete_utf8_decodes_to_the_replacement_character() {
    let tk = HfTokenizer::from_vocabulary(gpt2_vocab(None)).unwrap();
    let ids = tk.encode("é").unwrap();
    assert_eq!(ids.len(), 2, "two bytes, two byte tokens");
    let partial = tk.decode(&ids[..1]).unwrap();
    assert!(partial.ends_with('\u{FFFD}'), "{partial:?}");
    assert_eq!(tk.decode(&ids).unwrap(), "é");
}

#[test]
fn every_named_pre_tokenizer_builds() {
    for pre in [
        None,
        Some("llama-bpe"),
        Some("qwen2"),
        Some("smollm"),
        Some("default"),
        Some("something-new"),
    ] {
        let tk = HfTokenizer::from_vocabulary(gpt2_vocab(pre)).unwrap();
        assert_eq!(
            tk.decode(&tk.encode("It's 2024, hello world!").unwrap())
                .unwrap(),
            "It's 2024, hello world!",
            "{pre:?}"
        );
    }
    assert_eq!(pre_tokenizer_pattern(Some("qwen2")), QWEN2_PATTERN);
    assert_eq!(pre_tokenizer_pattern(Some("llama-bpe")), LLAMA3_PATTERN);
    assert_eq!(pre_tokenizer_pattern(None), GPT2_PATTERN);
}

/// A SentencePiece vocabulary built by hand: unk, bos, eos, a few words with
/// scores, and the 256 byte tokens that make byte fallback possible.
fn llama_vocab() -> Vocabulary {
    let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
    let mut scores = vec![0.0, 0.0, 0.0];
    let mut types = vec![2, TYPE_CONTROL, TYPE_CONTROL];
    for b in 0..=255u8 {
        tokens.push(format!("<0x{b:02X}>"));
        scores.push(-1000.0);
        types.push(6);
    }
    for (w, s) in [
        ("▁hello", -1.0),
        ("▁world", -2.0),
        ("▁", -3.0),
        ("hello", -4.0),
        ("▁there", -2.5),
    ] {
        tokens.push(w.into());
        scores.push(s);
        types.push(1);
    }
    Vocabulary {
        model: "llama".into(),
        tokens,
        scores,
        token_types: types,
        bos_token_id: Some(1),
        eos_token_ids: vec![2],
        unk_token_id: Some(0),
        add_bos_token: None,
        ..Default::default()
    }
}

#[test]
fn sentencepiece_uses_scores_and_byte_fallback() {
    let tk = HfTokenizer::from_vocabulary(llama_vocab()).unwrap();
    let ids = tk.encode("hello world").unwrap();
    let hello = tk.vocab.tokens.iter().position(|t| t == "▁hello").unwrap() as u32;
    let world = tk.vocab.tokens.iter().position(|t| t == "▁world").unwrap() as u32;
    assert_eq!(ids, vec![hello, world]);
    assert_eq!(tk.decode(&ids).unwrap(), "hello world");
    assert!(
        tk.add_bos(),
        "SentencePiece checkpoints want a BOS unless they say otherwise"
    );
    assert_eq!(tk.bos_token(), Some(1));
    assert_eq!(tk.special_token_text(1).as_deref(), Some("<s>"));

    let ids = tk.encode("hi").unwrap();
    let space = tk.vocab.tokens.iter().position(|t| t == "▁").unwrap() as u32;
    assert_eq!(
        ids[0], space,
        "the prepended word boundary is its own token"
    );
    assert!(
        ids[1..].iter().all(|&i| (3..259).contains(&i)),
        "an unknown word goes to byte tokens: {ids:?}"
    );
    assert_eq!(ids.len(), 3);
    assert_eq!(tk.decode(&ids).unwrap(), "hi");

    let ids = tk.encode("<s>hello there</s>").unwrap();
    assert_eq!(ids.first(), Some(&1));
    assert_eq!(ids.last(), Some(&2));
    assert_eq!(tk.decode(&ids).unwrap(), "hello there");
}

#[test]
fn unsupported_vocabulary_types_say_so() {
    let v = Vocabulary {
        model: "rwkv".into(),
        tokens: vec!["a".into()],
        ..Default::default()
    };
    match HfTokenizer::from_vocabulary(v) {
        Err(Error::Invalid(m)) => assert!(m.contains("rwkv") && m.contains("gpt2"), "{m}"),
        _ => panic!("expected Invalid"),
    }
    let mut v = llama_vocab();
    v.scores.pop();
    assert!(
        matches!(HfTokenizer::from_vocabulary(v), Err(Error::Invalid(m)) if m.contains("scores"))
    );
}
