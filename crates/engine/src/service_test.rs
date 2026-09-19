use super::*;
use crate::deps::memory::{ByteTokenizer, MemoryClock, MemoryModel};
use crate::logic::CHATML_STOP;

fn engine(model: MemoryModel, tokenizer: ByteTokenizer) -> Engine {
    Engine::new(
        Box::new(model),
        Box::new(tokenizer),
        Box::new(MemoryClock::default()),
    )
}

fn run(e: &Engine, prompt: &str, params: SamplingParams) -> (Completion, Vec<String>) {
    let mut chunks = Vec::new();
    let c = e
        .generate(
            &Cancel::new(),
            GenerateRequest {
                prompt: prompt.into(),
                params,
                raw: false,
            },
            &mut |s| chunks.push(s.to_string()),
        )
        .unwrap();
    (c, chunks)
}

#[test]
fn generates_until_eos_and_reports_usage_timing_and_the_prompt() {
    let e = engine(MemoryModel::reply("Hi there"), ByteTokenizer::new());
    let (c, chunks) = run(&e, "Q: hi\nA:", SamplingParams::greedy());
    assert_eq!(c.text, "Hi there");
    assert_eq!(chunks.concat(), "Hi there");
    assert_eq!(c.finish, FinishReason::Stop);
    assert_eq!(
        c.usage,
        Usage {
            prompt_tokens: 8,
            completion_tokens: 8,
            cached_tokens: 0
        }
    );
    assert_eq!(c.prompt, "Q: hi\nA:");
    assert_eq!(c.model, "memory");
    assert!(c.timing.prefill > Duration::ZERO);
    assert!(c.timing.time_to_first_token > c.timing.prefill);
    assert!(c.timing.total > c.timing.time_to_first_token);
    assert!(c.timing.decode > Duration::ZERO);
}

#[test]
fn max_tokens_and_the_context_window_both_end_with_length() {
    let e = engine(MemoryModel::cycle("abc"), ByteTokenizer::new());
    let (c, _) = run(
        &e,
        "x",
        SamplingParams {
            max_tokens: 5,
            ..SamplingParams::greedy()
        },
    );
    assert_eq!(c.text, "abcab");
    assert_eq!(c.finish, FinishReason::Length);
    assert_eq!(c.usage.completion_tokens, 5);

    let e = engine(
        MemoryModel::cycle("abc").with_context_length(6),
        ByteTokenizer::new(),
    );
    let (c, _) = run(
        &e,
        "xy",
        SamplingParams {
            max_tokens: 100,
            ..SamplingParams::greedy()
        },
    );
    assert_eq!(
        c.text, "abca",
        "2 prompt + 4 generated fills a context of 6"
    );
    assert_eq!(c.finish, FinishReason::Length);
}

#[test]
fn stop_sequences_are_held_back_across_tokens_and_never_leak() {
    let e = engine(MemoryModel::cycle("hello world "), ByteTokenizer::new());
    let params = SamplingParams {
        stop: vec!["wor".into()],
        max_tokens: 100,
        ..SamplingParams::greedy()
    };
    let (c, chunks) = run(&e, "say:", params);
    assert_eq!(c.text, "hello ");
    assert_eq!(c.finish, FinishReason::StopSequence);
    assert!(chunks.iter().all(|s| !s.contains('w')), "{chunks:?}");
    // Every byte is its own token here, so the partial 'w', 'wo' were held.
    assert_eq!(c.usage.completion_tokens, 9);
}

#[test]
fn cancellation_keeps_what_was_produced() {
    let e = engine(MemoryModel::cycle("abcdef"), ByteTokenizer::new());
    let cancel = Cancel::new();
    let inner = cancel.clone();
    let mut seen = String::new();
    let c = e
        .generate(
            &cancel,
            GenerateRequest {
                prompt: "go".into(),
                params: SamplingParams {
                    max_tokens: 100,
                    ..SamplingParams::greedy()
                },
                raw: false,
            },
            &mut |s| {
                seen.push_str(s);
                if seen.len() >= 3 {
                    inner.cancel();
                }
            },
        )
        .unwrap();
    assert_eq!(c.finish, FinishReason::Cancelled);
    assert_eq!(c.text, "abc");
    assert_eq!(seen, "abc");

    let pre = Cancel::new();
    pre.cancel();
    let err = e
        .generate(
            &pre,
            GenerateRequest {
                prompt: "go".into(),
                params: SamplingParams::greedy(),
                raw: false,
            },
            &mut |_| {},
        )
        .unwrap_err();
    assert!(err.is_cancelled());
}

#[test]
fn chat_renders_chatml_by_default_and_reuses_the_cached_prefix() {
    let model = MemoryModel::reply("Hello!");
    let e = engine(model, ByteTokenizer::new());
    let first = vec![Message::user("Hi")];
    let mut c = e
        .chat(
            &Cancel::new(),
            ChatRequest {
                messages: first.clone(),
                params: SamplingParams::greedy(),
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(c.text, "Hello!");
    assert_eq!(
        c.prompt,
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(c.prompt, e.render(&first).unwrap());
    assert_eq!(c.usage.cached_tokens, 0);
    let first_prompt_tokens = c.usage.prompt_tokens;

    let mut second = first.clone();
    second.push(Message::assistant(c.text.clone()));
    second.push(Message::user("Again"));
    c = e
        .chat(
            &Cancel::new(),
            ChatRequest {
                messages: second,
                params: SamplingParams::greedy(),
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(c.text, "Hello!");
    // The whole first prompt plus the generated reply were already in the
    // cache (every reply token was fed, because EOS was sampled on the step
    // after the last one); only "<|im_end|>\n<|im_start|>user\nAgain..." had
    // to be fed.
    assert_eq!(c.usage.cached_tokens, first_prompt_tokens + "Hello!".len());
    assert!(c.usage.cached_tokens > 0);
    let rendered = e
        .render(&[
            Message::user("Hi"),
            Message::assistant("Hello!"),
            Message::user("Again"),
        ])
        .unwrap();
    assert_eq!(c.usage.prompt_tokens, rendered.len());
}

#[test]
fn the_prompt_cache_feeds_only_the_new_tail_and_rewinds_on_divergence() {
    let model = MemoryModel::cycle("z");
    let feeds = model.feed_log();
    let e = engine(model, ByteTokenizer::new());
    let p = SamplingParams {
        max_tokens: 2,
        ..SamplingParams::greedy()
    };
    run(&e, "abcd", p.clone());
    assert_eq!(
        feeds.lock().unwrap().as_slice(),
        &[b"abcd".map(u32::from).to_vec(), vec![b'z' as u32]]
    );

    feeds.lock().unwrap().clear();
    let (c, _) = run(&e, "abcdzzq", p.clone());
    // Cache held a,b,c,d,z (the second z was sampled but never fed). Prompt
    // shares a,b,c,d,z; feed "zq".
    assert_eq!(c.usage.cached_tokens, 5);
    assert_eq!(feeds.lock().unwrap()[0], vec![b'z' as u32, b'q' as u32]);

    feeds.lock().unwrap().clear();
    let (c, _) = run(&e, "abXY", p);
    assert_eq!(c.usage.cached_tokens, 2, "diverged after ab");
    assert_eq!(feeds.lock().unwrap()[0], vec![b'X' as u32, b'Y' as u32]);

    e.reset_cache();
    feeds.lock().unwrap().clear();
    let (c, _) = run(
        &e,
        "abXY",
        SamplingParams {
            max_tokens: 1,
            ..SamplingParams::greedy()
        },
    );
    assert_eq!(c.usage.cached_tokens, 0);
}

#[test]
fn an_identical_prompt_still_feeds_one_token() {
    let e = engine(MemoryModel::cycle("z"), ByteTokenizer::new());
    let p = SamplingParams {
        max_tokens: 1,
        ..SamplingParams::greedy()
    };
    run(&e, "same", p.clone());
    let (c, _) = run(&e, "same", p);
    assert_eq!(c.usage.cached_tokens, 3);
    assert_eq!(c.text, "z");
}

#[test]
fn bos_is_added_once_unless_raw() {
    let model = MemoryModel::reply("ok");
    let feeds = model.feed_log();
    let e = engine(model, ByteTokenizer::new().with_add_bos(true));
    let (c, _) = run(&e, "hi", SamplingParams::greedy());
    assert_eq!(feeds.lock().unwrap()[0][0], ByteTokenizer::BOS);
    assert_eq!(c.usage.prompt_tokens, 3);

    e.reset_cache();
    feeds.lock().unwrap().clear();
    let (c, _) = run(&e, "<s>hi", SamplingParams::greedy());
    assert_ne!(
        feeds.lock().unwrap()[0][0],
        ByteTokenizer::BOS,
        "the prompt already starts with the BOS text"
    );
    assert_eq!(c.usage.prompt_tokens, 5);

    e.reset_cache();
    feeds.lock().unwrap().clear();
    e.generate(
        &Cancel::new(),
        GenerateRequest {
            prompt: "hi".into(),
            params: SamplingParams::greedy(),
            raw: true,
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(feeds.lock().unwrap()[0], vec![b'h' as u32, b'i' as u32]);
}

#[test]
fn a_custom_template_is_used_and_chatml_stop_is_only_added_for_the_fallback() {
    let t = "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}[assistant] ";
    let model = MemoryModel::reply("yo");
    let e = engine(model, ByteTokenizer::new().with_chat_template(t));
    let c = e
        .chat(
            &Cancel::new(),
            ChatRequest {
                messages: vec![Message::user("q")],
                params: SamplingParams::greedy(),
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(c.prompt, "[user] q\n[assistant] ");
    assert_eq!(c.text, "yo");

    // With the fallback, "<|im_end|>" as text stops generation even when
    // the model never emits an EOS token.
    let e = engine(MemoryModel::cycle("a<|im_end|>b"), ByteTokenizer::new());
    let c = e
        .chat(
            &Cancel::new(),
            ChatRequest {
                messages: vec![Message::user("q")],
                params: SamplingParams {
                    max_tokens: 50,
                    ..SamplingParams::greedy()
                },
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(c.text, "a");
    assert_eq!(c.finish, FinishReason::StopSequence);
    assert_eq!(CHATML_STOP, "<|im_end|>");
}

#[test]
fn multibyte_characters_split_across_tokens_are_emitted_whole() {
    let e = engine(MemoryModel::reply("héllo 世界"), ByteTokenizer::new());
    let (c, chunks) = run(&e, "p", SamplingParams::greedy());
    assert_eq!(c.text, "héllo 世界");
    assert!(chunks.iter().all(|s| !s.contains('\u{FFFD}')), "{chunks:?}");
    assert!(chunks.iter().any(|s| s == "é"), "{chunks:?}");
    assert_eq!(c.usage.completion_tokens, "héllo 世界".len());
}

#[test]
fn failures_are_specific() {
    let e = engine(
        MemoryModel::cycle("a").with_context_length(4),
        ByteTokenizer::new(),
    );
    match e.generate(
        &Cancel::new(),
        GenerateRequest {
            prompt: "toolong".into(),
            params: SamplingParams::greedy(),
            raw: false,
        },
        &mut |_| {},
    ) {
        Err(Error::Invalid(m)) => assert!(m.contains("7 tokens") && m.contains("4"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        e.generate(&Cancel::new(), GenerateRequest { prompt: "".into(), params: SamplingParams::greedy(), raw: false }, &mut |_| {}),
        Err(Error::Invalid(m)) if m.contains("empty")
    ));
    assert!(matches!(
        e.generate(&Cancel::new(), GenerateRequest { prompt: "x".into(), params: SamplingParams { top_p: 2.0, ..Default::default() }, raw: false }, &mut |_| {}),
        Err(Error::Invalid(m)) if m.contains("top_p")
    ));
    assert!(matches!(
        e.chat(
            &Cancel::new(),
            ChatRequest {
                messages: vec![],
                params: SamplingParams::greedy()
            },
            &mut |_| {}
        ),
        Err(Error::Invalid(_))
    ));
    assert_eq!(e.context_length(), 4);
    assert_eq!(e.model_id(), "memory");
}

#[test]
fn seeded_sampling_is_reproducible() {
    let e = engine(MemoryModel::cycle("abc"), ByteTokenizer::new());
    let p = SamplingParams {
        seed: Some(7),
        temperature: 1.0,
        max_tokens: 6,
        ..Default::default()
    };
    let (a, _) = run(&e, "s", p.clone());
    let (b, _) = run(&e, "s", p);
    assert_eq!(a.text, b.text);
    assert_eq!(a.text, "abcabc");
}
