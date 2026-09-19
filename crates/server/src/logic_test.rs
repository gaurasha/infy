use super::*;
use infy_kernel::Error;

#[test]
fn frames_have_the_openai_shape() {
    let f = serde_json::to_value(first_chunk("id", 1, "m")).unwrap();
    assert_eq!(f["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(f["choices"][0]["delta"]["content"], "");
    assert!(f["choices"][0].get("finish_reason").is_none());

    let d = serde_json::to_value(delta_chunk("id", 1, "m", "hi")).unwrap();
    assert_eq!(d["choices"][0]["delta"]["content"], "hi");
    assert!(d["choices"][0]["delta"].get("role").is_none());

    let l = serde_json::to_value(final_chunk(
        "id",
        1,
        "m",
        FinishReason::Length,
        Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            cached_tokens: 0,
        },
    ))
    .unwrap();
    assert_eq!(l["choices"][0]["finish_reason"], "length");
    assert_eq!(l["usage"]["total_tokens"], 3);
    assert_eq!(l["object"], "chat.completion.chunk");
}

#[test]
fn errors_map_to_statuses() {
    assert_eq!(error_status(&Error::invalid("x")).0, 400);
    assert_eq!(error_status(&Error::not_found("x")).0, 404);
    assert_eq!(error_status(&Error::unavailable("x")).0, 503);
    assert_eq!(error_status(&Error::internal("x")).0, 500);
    assert_eq!(
        error_status(&Error::invalid("x")).1.error.message,
        "invalid: x"
    );
}

#[test]
fn requested_model_ignores_blank() {
    assert_eq!(requested_model(None), None);
    assert_eq!(requested_model(Some("  ")), None);
    assert_eq!(requested_model(Some(" m ")), Some("m"));
}
