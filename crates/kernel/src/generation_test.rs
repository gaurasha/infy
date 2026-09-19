use super::*;
use crate::{Cancel, Role};

#[test]
fn defaults_validate() {
    SamplingParams::default().validate().unwrap();
    SamplingParams::greedy().validate().unwrap();
}

#[test]
fn validation_names_the_field() {
    let cases: Vec<(SamplingParams, &str)> = vec![
        (
            SamplingParams {
                max_tokens: 0,
                ..Default::default()
            },
            "max_tokens",
        ),
        (
            SamplingParams {
                temperature: -0.1,
                ..Default::default()
            },
            "temperature",
        ),
        (
            SamplingParams {
                temperature: f32::NAN,
                ..Default::default()
            },
            "temperature",
        ),
        (
            SamplingParams {
                top_p: 0.0,
                ..Default::default()
            },
            "top_p",
        ),
        (
            SamplingParams {
                top_p: 1.5,
                ..Default::default()
            },
            "top_p",
        ),
        (
            SamplingParams {
                repetition_penalty: 0.0,
                ..Default::default()
            },
            "repetition_penalty",
        ),
        (
            SamplingParams {
                stop: vec![String::new()],
                ..Default::default()
            },
            "stop",
        ),
    ];
    for (p, field) in cases {
        match p.validate() {
            Err(Error::Invalid(m)) => assert!(m.contains(field), "{m} should mention {field}"),
            other => panic!("expected Invalid for {field}, got {other:?}"),
        }
    }
}

#[test]
fn rates_never_divide_by_zero() {
    let t = Timing::default();
    assert_eq!(t.decode_tokens_per_second(10), 0.0);
    assert_eq!(t.prefill_tokens_per_second(10), 0.0);
    let t = Timing {
        decode: Duration::from_secs(2),
        ..Default::default()
    };
    // 11 tokens: the first is produced by prefill, ten by decode.
    assert_eq!(t.decode_tokens_per_second(11), 5.0);
    assert_eq!(t.decode_tokens_per_second(0), 0.0);
}

#[test]
fn finish_reason_wire_names_are_snake_case() {
    assert_eq!(
        serde_json::to_string(&FinishReason::StopSequence).unwrap(),
        "\"stop_sequence\""
    );
}

#[test]
fn role_round_trips_through_its_string_form() {
    for r in [Role::System, Role::User, Role::Assistant] {
        assert_eq!(r.as_str().parse::<Role>().unwrap(), r);
    }
    assert!(matches!("tool".parse::<Role>(), Err(Error::Invalid(_))));
}

#[test]
fn error_display_carries_context_and_source() {
    let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
    let e = Error::wrap("opening model.gguf", io);
    assert_eq!(e.to_string(), "opening model.gguf: gone");
    assert!(std::error::Error::source(&e).is_some());
    assert_eq!(Error::internal("plain").to_string(), "plain");
    assert!(Error::Cancelled.is_cancelled());
}

#[test]
fn cancel_flag_is_shared_between_clones() {
    let c = Cancel::new();
    let c2 = c.clone();
    assert!(c.check().is_ok());
    c2.cancel();
    assert!(c.is_cancelled());
    assert!(c.check().unwrap_err().is_cancelled());
}
