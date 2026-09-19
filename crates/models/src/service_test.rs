use super::*;
use crate::deps::gguf::{write_gguf, GgufFile};
use crate::deps::memory::RandomModel;
use crate::logic::{vocabulary_from_metadata, Architecture, MetaValue};
use infy_kernel::Vocabulary;

fn tiny() -> Llama {
    load(&RandomModel::tiny(7), &LoadOptions::default()).unwrap()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(worst <= tol, "{what}: max abs diff {worst} > {tol}");
}

const TOKENS: &[u32] = &[3, 1, 4, 1, 5, 9, 2, 6, 5, 3];

#[test]
fn a_tiny_model_loads_and_produces_finite_logits() {
    let m = tiny();
    assert_eq!(m.config().vocab_size, 64);
    let mut cache = m.new_cache();
    let logits = m.forward(&[1, 2, 3], &mut cache).unwrap();
    assert_eq!(logits.len(), 64);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(cache.len(), 3);
    // Not degenerate: different positions prefer different tokens.
    let all = m.forward_all(TOKENS, &mut m.new_cache()).unwrap();
    let argmax = |v: &Vec<f32>| {
        v.iter()
            .enumerate()
            .fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
    };
    let preds: Vec<usize> = all.iter().map(argmax).collect();
    assert!(preds.iter().any(|&p| p != preds[0]), "{preds:?}");
}

/// THE cache test. The logits for position i must not depend on whether the
/// tokens before it were fed all at once, one at a time, or in chunks. If
/// RoPE offsets, the causal mask, or the cache's bookkeeping disagree, this
/// is where it shows.
#[test]
fn incremental_feeding_matches_full_prefill() {
    let m = tiny();
    let full = m.forward_all(TOKENS, &mut m.new_cache()).unwrap();

    let mut cache = m.new_cache();
    for (i, &tok) in TOKENS.iter().enumerate() {
        let step = m.forward(&[tok], &mut cache).unwrap();
        assert_close(
            &step,
            &full[i],
            1e-4,
            &format!("one-at-a-time, position {i}"),
        );
        assert_eq!(cache.len(), i + 1);
    }

    let mut cache = m.new_cache();
    let chunks: &[&[u32]] = &[&TOKENS[..3], &TOKENS[3..4], &TOKENS[4..9], &TOKENS[9..]];
    let mut pos = 0;
    for chunk in chunks {
        let all = m.forward_all(chunk, &mut cache).unwrap();
        for (j, logits) in all.iter().enumerate() {
            assert_close(
                logits,
                &full[pos + j],
                1e-4,
                &format!("chunked, position {}", pos + j),
            );
        }
        pos += chunk.len();
    }
    assert_eq!(cache.len(), TOKENS.len());
}

#[test]
fn truncate_rewinds_and_refeeding_reproduces_the_same_logits() {
    let m = tiny();
    let full = m.forward_all(TOKENS, &mut m.new_cache()).unwrap();
    let mut cache = m.new_cache();
    m.forward(&TOKENS[..6], &mut cache).unwrap();
    // A different continuation, then rewind to the common prefix.
    m.forward(&[7, 7, 7], &mut cache).unwrap();
    assert_eq!(cache.len(), 9);
    cache.truncate(4);
    assert_eq!(cache.len(), 4);
    let all = m.forward_all(&TOKENS[4..], &mut cache).unwrap();
    for (j, logits) in all.iter().enumerate() {
        assert_close(
            logits,
            &full[4 + j],
            1e-4,
            &format!("after truncate, position {}", 4 + j),
        );
    }
    cache.reset();
    assert!(cache.is_empty());
    let again = m.forward(&TOKENS[..1], &mut cache).unwrap();
    assert_close(&again, &full[0], 1e-4, "after reset");
}

#[test]
fn cache_grows_by_doubling_and_keeps_what_it_had() {
    let m = tiny();
    let mut cache = m.new_cache();
    assert_eq!(cache.capacity(), 0);
    let prompt: Vec<u32> = (0..100).map(|i| (i * 7 % 64) as u32).collect();
    let full = m.forward_all(&prompt, &mut m.new_cache()).unwrap();
    m.forward(&prompt[..60], &mut cache).unwrap();
    assert_eq!(cache.capacity(), 64);
    let steps = m.forward_all(&prompt[60..66], &mut cache).unwrap();
    assert_eq!(cache.capacity(), 128, "doubled when position 65 arrived");
    for (j, step) in steps.iter().enumerate() {
        assert_close(
            step,
            &full[60 + j],
            1e-4,
            &format!("logits across the growth boundary, {}", 60 + j),
        );
    }
    let rest = m.forward_all(&prompt[66..], &mut cache).unwrap();
    assert_close(
        rest.last().unwrap(),
        full.last().unwrap(),
        1e-4,
        "last logits after growth",
    );
    assert_eq!(cache.len(), 100);
    assert_eq!(
        cache.memory_bytes(),
        2 * 2 * 128 * 2 * 8 * 4,
        "layers × (k,v) × cap × kv heads × dim × f32"
    );
}

#[test]
fn qwen2_style_biases_and_neox_rope_pass_the_same_invariant() {
    let cfg = RandomModel::tiny_config(Architecture::Qwen2);
    let src = RandomModel::new(&cfg, 11, candle_core::quantized::GgmlDType::Q4_0);
    assert!(src.has_tensor("blk.0.attn_q.bias"));
    let m = load(&src, &LoadOptions::default()).unwrap();
    assert!(!m.config().rope_interleaved());
    let full = m.forward_all(TOKENS, &mut m.new_cache()).unwrap();
    let mut cache = m.new_cache();
    for (i, &tok) in TOKENS.iter().enumerate() {
        let step = m.forward(&[tok], &mut cache).unwrap();
        assert_close(&step, &full[i], 1e-4, &format!("qwen2 position {i}"));
    }
}

#[test]
fn gguf_round_trip_is_lossless() {
    let vocab = Vocabulary {
        model: "gpt2".into(),
        pre: Some("default".into()),
        tokens: (0..64).map(|i| format!("t{i}")).collect(),
        token_types: vec![1; 64],
        merges: vec!["t1 t2".into()],
        bos_token_id: Some(0),
        eos_token_ids: vec![1],
        chat_template: Some("{{ bos_token }}".into()),
        ..Default::default()
    };
    let src = RandomModel::tiny(7).with_vocabulary(&vocab);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    write_gguf(&path, &src).unwrap();
    assert!(std::fs::metadata(&path).unwrap().len() > 1000);

    let file = GgufFile::open(&path).unwrap();
    assert_eq!(file.tensor_names(), src.tensor_names());
    assert!(file.tensor_bytes() > 0);
    assert_eq!(vocabulary_from_metadata(file.metadata()).unwrap(), vocab);
    assert_eq!(
        ModelConfig::from_metadata(file.metadata()).unwrap(),
        ModelConfig::from_metadata(src.metadata()).unwrap()
    );

    let a = load(&src, &LoadOptions::default()).unwrap();
    let b = load(&file, &LoadOptions::default()).unwrap();
    let la = a.forward_all(TOKENS, &mut a.new_cache()).unwrap();
    let lb = b.forward_all(TOKENS, &mut b.new_cache()).unwrap();
    for (i, (x, y)) in la.iter().zip(&lb).enumerate() {
        assert_close(x, y, 1e-6, &format!("round trip position {i}"));
    }
}

#[test]
fn failures_are_specific() {
    let m = tiny();
    let mut cache = m.new_cache();
    let too_long: Vec<u32> = vec![1; 300];
    match m.forward(&too_long, &mut cache) {
        Err(infy_kernel::Error::Invalid(msg)) => {
            assert!(msg.contains("300") && msg.contains("256"), "{msg}")
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        m.forward(&[], &mut cache),
        Err(infy_kernel::Error::Invalid(_))
    ));
    assert!(
        matches!(m.forward(&[64], &mut cache), Err(infy_kernel::Error::Invalid(msg)) if msg.contains("64"))
    );

    let bad =
        RandomModel::tiny(1).with_metadata("general.architecture", MetaValue::Str("phi3".into()));
    assert!(
        matches!(load(&bad, &LoadOptions::default()), Err(infy_kernel::Error::Invalid(msg)) if msg.contains("phi3"))
    );

    match GgufFile::open("/nonexistent/model.gguf") {
        Err(infy_kernel::Error::NotFound(msg)) => assert!(msg.contains("model.gguf")),
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("opened a file that does not exist"),
    }
    assert!(device("cpu").unwrap().is_cpu());
    assert!(matches!(device("tpu"), Err(infy_kernel::Error::Invalid(_))));
    assert!(matches!(
        device("cuda:x"),
        Err(infy_kernel::Error::Invalid(_))
    ));
}

#[test]
fn a_checkpoint_missing_tensors_says_which() {
    let cfg = RandomModel::tiny_config(Architecture::Llama);
    let mut big = cfg.clone();
    big.num_layers = 3;
    // Metadata claims three layers; weights exist for two.
    let src = RandomModel::new(&cfg, 3, candle_core::quantized::GgmlDType::Q8_0)
        .with_metadata("llama.block_count", MetaValue::U32(3));
    match load(&src, &LoadOptions::default()) {
        Err(infy_kernel::Error::Invalid(msg)) => assert!(msg.contains("blk.2."), "{msg}"),
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("loaded a checkpoint with missing tensors"),
    }
}
