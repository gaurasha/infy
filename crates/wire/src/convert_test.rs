use super::*;
use crate::chat::{ChatMessage, Content, ContentPart, Stop};
use crate::completion::Prompt;
use crate::{code_for, data_line, error_response, frame, Code};
use infy_kernel::{Role, Timing};

fn req(json: &str) -> ChatCompletionRequest {
    serde_json::from_str(json).unwrap()
}

#[test]
fn a_minimal_openai_request_parses_and_takes_defaults() {
    let r = req(r#"{"model":"anything","messages":[{"role":"user","content":"hi"}]}"#);
    let (msgs, p) = chat_request_to_kernel(&r, &SamplingParams::default()).unwrap();
    assert_eq!(msgs, vec![Message::user("hi")]);
    assert_eq!(p, SamplingParams::default());
}

#[test]
fn client_parameters_override_defaults_and_are_validated() {
    let r = req(
        r#"{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],
            "max_tokens":7,"temperature":0,"top_p":0.5,"top_k":3,"stop":"END","seed":42,
            "repetition_penalty":1.2}"#,
    );
    let (msgs, p) = chat_request_to_kernel(&r, &SamplingParams::default()).unwrap();
    assert_eq!(msgs[0].role, Role::System);
    assert_eq!(p.max_tokens, 7);
    assert_eq!(p.temperature, 0.0);
    assert_eq!(p.top_p, 0.5);
    assert_eq!(p.top_k, 3);
    assert_eq!(p.stop, vec!["END".to_string()]);
    assert_eq!(p.seed, Some(42));
    assert_eq!(p.repetition_penalty, 1.2);

    let bad = req(r#"{"messages":[{"role":"user","content":"hi"}],"temperature":-1}"#);
    assert!(matches!(
        chat_request_to_kernel(&bad, &SamplingParams::default()),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn content_parts_and_stop_lists_are_accepted() {
    let r = req(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url"},{"type":"text","text":"b"}]}],
            "stop":["x","y"],"max_completion_tokens":3}"#,
    );
    assert_eq!(
        r.messages[0].content,
        Some(Content::Parts(vec![
            ContentPart {
                kind: "text".into(),
                text: Some("a".into())
            },
            ContentPart {
                kind: "image_url".into(),
                text: None
            },
            ContentPart {
                kind: "text".into(),
                text: Some("b".into())
            },
        ]))
    );
    let (msgs, p) = chat_request_to_kernel(&r, &SamplingParams::default()).unwrap();
    assert_eq!(msgs[0].content, "ab");
    assert_eq!(p.stop, vec!["x".to_string(), "y".to_string()]);
    assert_eq!(p.max_tokens, 3);
    assert_eq!(Stop::One("z".into()).into_vec(), vec!["z".to_string()]);
}

#[test]
fn rejects_what_a_single_sequence_engine_cannot_do() {
    let empty = req(r#"{"messages":[]}"#);
    assert!(
        matches!(chat_request_to_kernel(&empty, &SamplingParams::default()), Err(Error::Invalid(m)) if m.contains("messages"))
    );
    let n2 = req(r#"{"messages":[{"role":"user","content":"hi"}],"n":2}"#);
    assert!(
        matches!(chat_request_to_kernel(&n2, &SamplingParams::default()), Err(Error::Invalid(m)) if m.contains("n must be 1"))
    );
    let tool = req(r#"{"messages":[{"role":"tool","content":"hi"}]}"#);
    assert!(
        matches!(chat_request_to_kernel(&tool, &SamplingParams::default()), Err(Error::Invalid(m)) if m.contains("tool"))
    );
}

#[test]
fn completion_prompt_forms() {
    let one: CompletionRequest = serde_json::from_str(r#"{"prompt":"abc"}"#).unwrap();
    let many: CompletionRequest = serde_json::from_str(r#"{"prompt":["ab","c"]}"#).unwrap();
    assert_eq!(one.prompt, Prompt::One("abc".into()));
    let (text, _) = completion_request_to_kernel(&many, &SamplingParams::default()).unwrap();
    assert_eq!(text, "abc");
}

fn completion() -> Completion {
    Completion {
        model: "tiny".into(),
        text: "hello".into(),
        finish: FinishReason::Stop,
        usage: Usage {
            prompt_tokens: 3,
            completion_tokens: 2,
            cached_tokens: 1,
        },
        timing: Timing::default(),
        prompt: "<prompt>".into(),
    }
}

#[test]
fn responses_have_the_openai_shape() {
    let r = chat_response("chatcmpl-1", 99, &completion());
    let v: serde_json::Value = serde_json::to_value(&r).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "hello");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["total_tokens"], 5);
    assert_eq!(v["created"], 99);

    let c = completion_response("cmpl-1", 1, &completion());
    let v: serde_json::Value = serde_json::to_value(&c).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "hello");
}

#[test]
fn chunks_omit_absent_fields_and_round_trip() {
    let first = chat_chunk(
        "id",
        1,
        "m",
        Delta {
            role: Some("assistant".into()),
            content: None,
        },
        None,
        None,
    );
    let s = serde_json::to_string(&first).unwrap();
    assert!(!s.contains("finish_reason"), "{s}");
    assert!(!s.contains("usage"), "{s}");
    assert!(!s.contains("\"content\""), "{s}");

    let last = chat_chunk(
        "id",
        1,
        "m",
        Delta::default(),
        Some(FinishReason::Length),
        Some(completion().usage),
    );
    let s = serde_json::to_string(&last).unwrap();
    let back: ChatCompletionChunk = serde_json::from_str(&s).unwrap();
    assert_eq!(back, last);
    assert_eq!(back.choices[0].finish_reason.as_deref(), Some("length"));
    assert_eq!(back.usage.unwrap().total_tokens, 5);

    let cc = completion_chunk("id", 1, "m", "tok", None, None);
    assert_eq!(cc.choices[0].text, "tok");
}

#[test]
fn a_real_ollama_chunk_decodes() {
    // Captured shape of an ollama /v1/chat/completions streaming frame.
    let line = r#"data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1700000000,"model":"llama3.2:3b","system_fingerprint":"fp_ollama","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}"#;
    let payload = data_line(line).unwrap();
    let chunk: ChatCompletionChunk = serde_json::from_str(payload).unwrap();
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    assert_eq!(chunk.choices[0].finish_reason, None);
}

#[test]
fn sse_framing() {
    assert_eq!(frame("{}"), "data: {}\n\n");
    assert_eq!(data_line("data: {\"a\":1}"), Some("{\"a\":1}"));
    assert_eq!(data_line("data: [DONE]"), None);
    assert_eq!(data_line(": keep-alive"), None);
    assert_eq!(data_line(""), None);
    assert_eq!(data_line("event: ping"), None);
}

#[test]
fn errors_map_to_codes_statuses_and_the_openai_envelope() {
    assert_eq!(code_for(&Error::invalid("x")), Code::BadRequest);
    assert_eq!(code_for(&Error::not_found("x")), Code::NotFound);
    assert_eq!(code_for(&Error::unavailable("x")), Code::Unavailable);
    assert_eq!(code_for(&Error::Cancelled), Code::Cancelled);
    assert_eq!(code_for(&Error::internal("x")), Code::Internal);
    assert_eq!(Code::BadRequest.http_status(), 400);
    assert_eq!(Code::Unavailable.http_status(), 503);

    let e = error_response(&Error::unavailable(
        "ollama is not running; start it with `infy runtime start ollama`",
    ));
    let v: serde_json::Value = serde_json::to_value(&e).unwrap();
    assert_eq!(v["error"]["type"], "service_unavailable_error");
    assert_eq!(v["error"]["code"], "unavailable");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("infy runtime start ollama"));
}

#[test]
fn chat_message_text_handles_missing_content() {
    let m = ChatMessage {
        role: "user".into(),
        content: None,
    };
    assert_eq!(m.text(), "");
    assert_eq!(ChatMessage::new("user", "x").text(), "x");
}
