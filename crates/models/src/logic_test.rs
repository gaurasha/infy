use super::*;

fn llama_meta() -> Metadata {
    let mut m = Metadata::new();
    let s = |v: &str| MetaValue::Str(v.into());
    m.insert("general.architecture".into(), s("llama"));
    m.insert("general.name".into(), s("SmolLM2 135M Instruct"));
    m.insert("llama.embedding_length".into(), MetaValue::U32(576));
    m.insert("llama.block_count".into(), MetaValue::U32(30));
    m.insert("llama.attention.head_count".into(), MetaValue::U32(9));
    m.insert("llama.attention.head_count_kv".into(), MetaValue::U32(3));
    m.insert("llama.feed_forward_length".into(), MetaValue::U32(1536));
    m.insert("llama.context_length".into(), MetaValue::U32(8192));
    m.insert(
        "llama.attention.layer_norm_rms_epsilon".into(),
        MetaValue::F32(1e-5),
    );
    m.insert("llama.rope.freq_base".into(), MetaValue::F32(100_000.0));
    m.insert("llama.vocab_size".into(), MetaValue::U32(49152));
    m
}

#[test]
fn reads_a_llama_cpp_style_header() {
    let c = ModelConfig::from_metadata(&llama_meta()).unwrap();
    assert_eq!(c.architecture, Architecture::Llama);
    assert_eq!(c.name, "SmolLM2 135M Instruct");
    assert_eq!(
        (c.hidden_size, c.num_layers, c.num_heads, c.num_kv_heads),
        (576, 30, 9, 3)
    );
    assert_eq!(
        c.head_dim, 64,
        "derived from hidden / heads when not stated"
    );
    assert_eq!(c.rope_theta, 100_000.0);
    assert_eq!(c.vocab_size, 49152);
    assert_eq!(c.rope_scaling, RopeScaling::None);
    assert!(
        c.rope_interleaved(),
        "llama.cpp permutes llama q/k for interleaved rope"
    );
}

#[test]
fn qwen2_uses_neox_rope_and_defaults_kv_heads_to_heads() {
    let mut m = Metadata::new();
    m.insert(
        "general.architecture".into(),
        MetaValue::Str("qwen2".into()),
    );
    m.insert("qwen2.embedding_length".into(), MetaValue::U64(896));
    m.insert("qwen2.block_count".into(), MetaValue::U32(24));
    m.insert("qwen2.attention.head_count".into(), MetaValue::U32(14));
    m.insert("qwen2.feed_forward_length".into(), MetaValue::U32(4864));
    m.insert("qwen2.context_length".into(), MetaValue::U32(32768));
    m.insert("qwen2.attention.key_length".into(), MetaValue::U32(64));
    m.insert(
        "tokenizer.ggml.tokens".into(),
        MetaValue::Array(vec![MetaValue::Str("a".into()), MetaValue::Str("b".into())]),
    );
    let c = ModelConfig::from_metadata(&m).unwrap();
    assert_eq!(c.architecture, Architecture::Qwen2);
    assert!(!c.rope_interleaved());
    assert_eq!(c.num_kv_heads, 14);
    assert_eq!(c.head_dim, 64);
    assert_eq!(c.vocab_size, 2, "falls back to the token list length");
    assert_eq!(c.rope_theta, 10_000.0, "default when absent");
    assert_eq!(c.name, "qwen2", "falls back to the architecture name");
}

#[test]
fn errors_name_the_missing_key_and_the_unsupported_architecture() {
    let mut m = llama_meta();
    m.remove("llama.block_count");
    match ModelConfig::from_metadata(&m) {
        Err(Error::Invalid(msg)) => assert!(msg.contains("llama.block_count"), "{msg}"),
        other => panic!("{other:?}"),
    }
    let mut m = llama_meta();
    m.insert(
        "general.architecture".into(),
        MetaValue::Str("gemma2".into()),
    );
    match ModelConfig::from_metadata(&m) {
        Err(Error::Invalid(msg)) => {
            assert!(
                msg.contains("gemma2") && msg.contains("llama, qwen2"),
                "{msg}"
            )
        }
        other => panic!("{other:?}"),
    }
    let mut m = llama_meta();
    m.insert(
        "llama.rope.scaling.type".into(),
        MetaValue::Str("yarn".into()),
    );
    assert!(
        matches!(ModelConfig::from_metadata(&m), Err(Error::Invalid(msg)) if msg.contains("yarn"))
    );
}

#[test]
fn linear_rope_scaling_is_read() {
    let mut m = llama_meta();
    m.insert(
        "llama.rope.scaling.type".into(),
        MetaValue::Str("linear".into()),
    );
    m.insert("llama.rope.scaling.factor".into(), MetaValue::F32(4.0));
    let c = ModelConfig::from_metadata(&m).unwrap();
    assert_eq!(c.rope_scaling, RopeScaling::Linear { factor: 4.0 });
}

#[test]
fn validation_rejects_impossible_shapes() {
    let mut c = ModelConfig::from_metadata(&llama_meta()).unwrap();
    c.num_kv_heads = 4;
    assert!(matches!(c.validate(), Err(Error::Invalid(m)) if m.contains("multiple")));
    c.num_kv_heads = 3;
    c.head_dim = 7;
    assert!(matches!(c.validate(), Err(Error::Invalid(m)) if m.contains("even")));
}

#[test]
fn metadata_round_trips_through_the_config() {
    let c = ModelConfig::from_metadata(&llama_meta()).unwrap();
    let back = ModelConfig::from_metadata(&c.to_metadata()).unwrap();
    assert_eq!(back, c);
    let mut lin = c.clone();
    lin.rope_scaling = RopeScaling::Linear { factor: 2.0 };
    assert_eq!(ModelConfig::from_metadata(&lin.to_metadata()).unwrap(), lin);
}

#[test]
fn tensor_spec_matches_what_llama_cpp_writes() {
    let c = ModelConfig::from_metadata(&llama_meta()).unwrap();
    let spec = c.tensor_spec();
    let find = |n: &str| {
        spec.iter()
            .find(|s| s.name == n)
            .unwrap_or_else(|| panic!("{n}"))
    };
    assert_eq!(find("token_embd.weight").shape, vec![49152, 576]);
    assert_eq!(find("blk.0.attn_q.weight").shape, vec![576, 576]);
    assert_eq!(
        find("blk.29.attn_k.weight").shape,
        vec![192, 576],
        "kv heads × head_dim rows"
    );
    assert_eq!(find("blk.0.attn_output.weight").shape, vec![576, 576]);
    assert_eq!(find("blk.0.ffn_gate.weight").shape, vec![1536, 576]);
    assert_eq!(find("blk.0.ffn_down.weight").shape, vec![576, 1536]);
    assert!(
        !find("output.weight").required,
        "tied embeddings are common"
    );
    assert!(!find("blk.0.attn_q.bias").required);
    assert!(find("blk.0.attn_q.weight").required);
    assert_eq!(spec.iter().filter(|s| s.required).count(), 2 + 30 * 9);

    let present: Vec<String> = spec
        .iter()
        .filter(|s| s.required)
        .map(|s| s.name.clone())
        .collect();
    assert!(c.missing_tensors(&present).is_empty());
    let missing = c.missing_tensors(&present[..present.len() - 1]);
    assert_eq!(missing, vec!["blk.29.ffn_down.weight".to_string()]);
}

#[test]
fn kv_cache_size_is_two_tensors_per_layer() {
    let c = ModelConfig::from_metadata(&llama_meta()).unwrap();
    // 30 layers × 2 × 3 kv heads × 64 dims × 4 bytes per position
    assert_eq!(c.kv_cache_bytes(1, 4), 30 * 2 * 3 * 64 * 4);
    assert_eq!(c.kv_cache_bytes(1000, 2), 30 * 2 * 3 * 64 * 2 * 1000);
}

#[test]
fn vocabulary_round_trips_and_collects_every_end_token() {
    let mut v = Vocabulary {
        model: "gpt2".into(),
        pre: Some("llama-bpe".into()),
        tokens: vec![
            "<|begin_of_text|>".into(),
            "a".into(),
            "b".into(),
            "<|eot_id|>".into(),
        ],
        scores: vec![],
        token_types: vec![3, 1, 1, 3],
        merges: vec!["a b".into()],
        bos_token_id: Some(0),
        eos_token_ids: vec![2, 3],
        unk_token_id: None,
        pad_token_id: Some(0),
        add_bos_token: Some(true),
        add_eos_token: Some(false),
        chat_template: Some("{{ messages }}".into()),
    };
    let back = vocabulary_from_metadata(&vocabulary_to_metadata(&v)).unwrap();
    assert_eq!(back, v);

    v.eos_token_ids = vec![2, 3, 5];
    assert_eq!(
        vocabulary_from_metadata(&vocabulary_to_metadata(&v))
            .unwrap()
            .eos_token_ids,
        vec![2, 3, 5]
    );

    let mut m = vocabulary_to_metadata(&v);
    m.insert("tokenizer.ggml.eot_token_id".into(), MetaValue::U32(2));
    assert_eq!(
        vocabulary_from_metadata(&m).unwrap().eos_token_ids,
        vec![2, 5],
        "duplicates collapse"
    );

    assert!(
        matches!(vocabulary_from_metadata(&Metadata::new()), Err(Error::Invalid(m)) if m.contains("tokenizer.ggml.tokens"))
    );
}

#[test]
fn rope_frequencies_follow_the_formula() {
    let f = rope_inv_freq(8, 10_000.0);
    let expect = [1.0, 0.1, 0.01, 0.001];
    for (a, b) in f.iter().zip(expect) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[test]
fn rope_tables_are_identity_at_zero_and_rotate_after() {
    let f = rope_inv_freq(4, 10_000.0);
    let (cos, sin) = rope_tables(&f, None, 0..2, &RopeScaling::None);
    assert_eq!(cos.len(), 4);
    assert_eq!(&cos[..2], &[1.0, 1.0]);
    assert_eq!(&sin[..2], &[0.0, 0.0]);
    assert!((cos[2] - 1f32.cos()).abs() < 1e-6);
    assert!((sin[3] - 0.01f32.sin()).abs() < 1e-6);

    // Linear scaling halves the angle; frequency factors divide the frequency.
    let (cos_lin, _) = rope_tables(&f, None, 2..3, &RopeScaling::Linear { factor: 2.0 });
    assert!((cos_lin[0] - 1f32.cos()).abs() < 1e-6);
    let (cos_ff, _) = rope_tables(&f, Some(&[2.0, 1.0]), 2..3, &RopeScaling::None);
    assert!((cos_ff[0] - 1f32.cos()).abs() < 1e-6);
    assert!((cos_ff[1] - 0.02f32.cos()).abs() < 1e-6);
}

#[test]
fn causal_mask_respects_the_cached_offset() {
    let m = causal_mask(2, 3);
    let inf = f32::NEG_INFINITY;
    assert_eq!(m, vec![0.0, 0.0, 0.0, 0.0, inf, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(causal_mask(1, 0), vec![0.0]);
    assert_eq!(
        causal_mask(3, 0),
        vec![0.0, inf, inf, 0.0, 0.0, inf, 0.0, 0.0, 0.0]
    );
}

#[test]
fn meta_value_widening() {
    assert_eq!(MetaValue::U8(7).as_u64(), Some(7));
    assert_eq!(MetaValue::I32(-1).as_u64(), None);
    assert_eq!(MetaValue::I64(9).as_f64(), Some(9.0));
    assert_eq!(MetaValue::Str("x".into()).as_u64(), None);
    assert_eq!(MetaValue::Bool(true).as_bool(), Some(true));
}
