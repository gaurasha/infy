use super::*;
use infy_kernel::Role;

fn greedy() -> SamplingParams {
    SamplingParams::greedy()
}

#[test]
fn greedy_is_argmax_and_ignores_the_random_number() {
    let logits = [0.1, 3.0, -1.0, 2.9];
    for u in [0.0, 0.5, 0.999] {
        assert_eq!(sample(&logits, &greedy(), &[], u), 1);
    }
    assert_eq!(sample(&[], &greedy(), &[], 0.5), 0);
    assert_eq!(
        sample(&[f32::NAN, 1.0, f32::INFINITY], &greedy(), &[], 0.5),
        1,
        "non-finite is ignored"
    );
}

#[test]
fn repetition_penalty_pushes_recent_tokens_down() {
    let logits = [2.0, 1.9, -1.0];
    let p = SamplingParams {
        repetition_penalty: 1.5,
        ..greedy()
    };
    assert_eq!(
        sample(&logits, &p, &[0], 0.0),
        1,
        "token 0 was recent: 2.0/1.5 < 1.9"
    );
    assert_eq!(sample(&logits, &p, &[1], 0.0), 0);
    let p = SamplingParams {
        repetition_penalty: 1.5,
        repetition_window: 1,
        ..greedy()
    };
    assert_eq!(
        sample(&logits, &p, &[0, 1], 0.0),
        0,
        "only the last token is in the window"
    );
    let negative = [-1.0, -1.2];
    assert_eq!(sample(&negative, &p, &[], 0.0), 0);
    assert_eq!(
        sample(&negative, &p, &[0], 0.0),
        1,
        "a negative logit is multiplied: -1.0 × 1.5 < -1.2"
    );
}

#[test]
fn top_k_one_is_greedy_and_top_k_two_samples_only_from_the_top_two() {
    let logits = [0.0, 5.0, 4.0, -10.0];
    let p = SamplingParams {
        temperature: 1.0,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    assert_eq!(sample(&logits, &p, &[], 0.99), 1);
    let p = SamplingParams { top_k: 2, ..p };
    let picks: std::collections::BTreeSet<u32> = (0..100)
        .map(|i| sample(&logits, &p, &[], i as f32 / 100.0))
        .collect();
    assert_eq!(picks, [1, 2].into_iter().collect());
}

#[test]
fn top_p_keeps_the_smallest_nucleus_reaching_p() {
    // softmax([4, 3, 0, 0]) ≈ [0.70, 0.26, 0.013, 0.013]
    let logits = [4.0, 3.0, 0.0, 0.0];
    let base = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let p = SamplingParams {
        top_p: 0.5,
        ..base.clone()
    };
    assert_eq!(
        sample(&logits, &p, &[], 0.999),
        0,
        "p=0.5 keeps only the top token"
    );
    let p = SamplingParams {
        top_p: 0.9,
        ..base.clone()
    };
    let picks: std::collections::BTreeSet<u32> = (0..50)
        .map(|i| sample(&logits, &p, &[], i as f32 / 50.0))
        .collect();
    assert_eq!(picks, [0, 1].into_iter().collect(), "p=0.9 keeps two");
    let p = SamplingParams { top_p: 1.0, ..base };
    assert_eq!(
        sample(&logits, &p, &[], 0.9999),
        3,
        "u near 1 picks the last candidate"
    );
    assert_eq!(sample(&logits, &p, &[], 0.0), 0);
}

#[test]
fn temperature_flattens_the_distribution() {
    let logits = [2.0, 0.0];
    let cold = SamplingParams {
        temperature: 0.1,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let hot = SamplingParams {
        temperature: 10.0,
        ..cold.clone()
    };
    // At u = 0.6: cold keeps token 0 (p ≈ 1.0); hot gives token 1 a ~45% share.
    assert_eq!(sample(&logits, &cold, &[], 0.6), 0);
    assert_eq!(sample(&logits, &hot, &[], 0.6), 1);
}

#[test]
fn rng_is_deterministic_and_in_range() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    let xs: Vec<f32> = (0..1000).map(|_| a.next_f32()).collect();
    let ys: Vec<f32> = (0..1000).map(|_| b.next_f32()).collect();
    assert_eq!(xs, ys);
    assert!(xs.iter().all(|&x| (0.0..1.0).contains(&x)));
    assert!(
        xs.iter().any(|&x| x < 0.1) && xs.iter().any(|&x| x > 0.9),
        "spread"
    );
    assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
    assert_ne!(Rng::new(0).next_u64(), 0, "the zero seed is remapped");
}

#[test]
fn stop_detection() {
    let stops = vec!["world".to_string(), "END".to_string()];
    assert_eq!(check_stop("hello ", &stops), StopCheck::NoMatch);
    assert_eq!(check_stop("hello wo", &stops), StopCheck::Partial(2));
    assert_eq!(check_stop("hello world!", &stops), StopCheck::Hit(6));
    assert_eq!(
        check_stop("xENDy world", &stops),
        StopCheck::Hit(1),
        "earliest wins"
    );
    assert_eq!(
        check_stop("héllo wo", &stops),
        StopCheck::Partial(2),
        "multi-byte text before the suffix"
    );
    assert_eq!(check_stop("", &stops), StopCheck::NoMatch);
    assert_eq!(check_stop("anything", &[]), StopCheck::NoMatch);
    let cjk = vec!["世界".to_string()];
    assert_eq!(
        check_stop("你好世", &cjk),
        StopCheck::Partial(3),
        "held back on a character boundary"
    );
    assert_eq!(check_stop("你好世界", &cjk), StopCheck::Hit(6));
}

#[test]
fn streaming_deltas() {
    assert_eq!(text_delta("hel", "hello"), Some("lo"));
    assert_eq!(text_delta("", "h"), Some("h"));
    assert_eq!(text_delta("hello", "hello"), Some(""));
    assert_eq!(
        text_delta("h", "h\u{FFFD}"),
        None,
        "wait for the rest of the character"
    );
    assert_eq!(
        text_delta("h\u{FFFD}", "hé"),
        Some("é"),
        "and emit it once complete"
    );
    assert_eq!(
        text_delta("abc", "abd"),
        Some("d"),
        "a rewritten prefix keeps the common part"
    );
    assert_eq!(text_delta("xyz", "abc"), Some("abc"));
}

#[test]
fn prefill_plans() {
    assert_eq!(
        plan_prefill(&[], &[1, 2, 3]),
        PrefillPlan {
            keep: 0,
            feed_from: 0
        }
    );
    assert_eq!(
        plan_prefill(&[1, 2, 3], &[1, 2, 3, 4, 5]),
        PrefillPlan {
            keep: 3,
            feed_from: 3
        }
    );
    assert_eq!(
        plan_prefill(&[1, 2, 9, 9], &[1, 2, 3]),
        PrefillPlan {
            keep: 2,
            feed_from: 2
        }
    );
    assert_eq!(
        plan_prefill(&[1, 2, 3], &[1, 2, 3]),
        PrefillPlan {
            keep: 2,
            feed_from: 2
        },
        "always feed at least one"
    );
    assert_eq!(
        plan_prefill(&[1, 2, 3, 4], &[1, 2]),
        PrefillPlan {
            keep: 1,
            feed_from: 1
        }
    );
    assert_eq!(
        plan_prefill(&[5], &[1]),
        PrefillPlan {
            keep: 0,
            feed_from: 0
        }
    );
    assert_eq!(
        plan_prefill(&[], &[]),
        PrefillPlan {
            keep: 0,
            feed_from: 0
        }
    );
}

#[test]
fn bos_rules() {
    assert!(needs_bos("hi", Some("<s>"), true));
    assert!(!needs_bos("<s>hi", Some("<s>"), true));
    assert!(!needs_bos("hi", Some("<s>"), false));
    assert!(needs_bos("hi", None, true));
    assert!(needs_bos("hi", Some(""), true));
}

fn msgs() -> Vec<Message> {
    vec![
        Message::system("Be brief."),
        Message::user("Hi "),
        Message::assistant("Hello!"),
        Message::user("Bye"),
    ]
}

const CHATML_TEMPLATE: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

#[test]
fn chatml_template_and_fallback_agree() {
    let rendered = render_chat_template(CHATML_TEMPLATE, &msgs(), None, None).unwrap();
    assert_eq!(rendered, chatml(&msgs()));
    assert_eq!(
        rendered,
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi <|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n"
    );
}

const LLAMA3_TEMPLATE: &str = "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

#[test]
fn llama3_style_template_uses_bos_and_trim() {
    let r = render_chat_template(
        LLAMA3_TEMPLATE,
        &msgs()[1..2],
        Some("<|begin_of_text|>"),
        Some("<|eot_id|>"),
    )
    .unwrap();
    assert_eq!(r, "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
}

const LLAMA2_TEMPLATE: &str = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}";

#[test]
fn llama2_style_template_uses_python_methods_slicing_and_raise_exception() {
    let r = render_chat_template(LLAMA2_TEMPLATE, &msgs(), Some("<s>"), Some("</s>")).unwrap();
    assert_eq!(
        r,
        "<s>[INST] <<SYS>>\nBe brief.\n<</SYS>>\n\nHi [/INST] Hello! </s><s>[INST] Bye [/INST]"
    );

    let bad = vec![Message::user("a"), Message::user("b")];
    match render_chat_template(LLAMA2_TEMPLATE, &bad, Some("<s>"), Some("</s>")) {
        Err(Error::Invalid(m)) => assert!(m.contains("must alternate"), "{m}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn template_with_namespace_loop_last_and_tools_none() {
    let t = "{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}{% if loop.last %}last={{ m.role }} {% endif %}{% endfor %}n={{ ns.n }}{% if tools is not none %} TOOLS{% endif %}";
    let r = render_chat_template(t, &msgs(), None, None).unwrap();
    assert_eq!(r, "last=user n=4");
    assert!(
        matches!(render_chat_template("{% for x in %}", &msgs(), None, None), Err(Error::Invalid(m)) if m.contains("parse"))
    );
    assert_eq!(msgs()[0].role, Role::System);
}
