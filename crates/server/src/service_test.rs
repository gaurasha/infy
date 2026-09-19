use super::*;
use crate::deps::memory::{MemoryBackend, MemoryClock, MemoryIds};
use axum::body::Body;
use axum::http::{header, Request};
use http_body_util::BodyExt;
use infy_wire::data_line;
use std::time::Duration;
use tower::ServiceExt;

fn app(backend: MemoryBackend) -> Router {
    let server = Server::new(
        Arc::new(backend),
        Arc::new(MemoryIds::default()),
        Arc::new(MemoryClock(1_700_000_000)),
        SamplingParams::default(),
    );
    router(Arc::new(server))
}

async fn call(app: Router, method: &str, path: &str, body: Option<&str>) -> (u16, String, String) {
    let mut req = Request::builder().method(method).uri(path);
    let body = match body {
        Some(b) => {
            req = req.header(header::CONTENT_TYPE, "application/json");
            Body::from(b.to_string())
        }
        None => Body::empty(),
    };
    let resp = app.oneshot(req.body(body).unwrap()).await.unwrap();
    let status = resp.status().as_u16();
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ctype, String::from_utf8_lossy(&bytes).into_owned())
}

fn json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap()
}

#[tokio::test]
async fn health_and_models() {
    let (status, _, body) = call(app(MemoryBackend::new("x")), "GET", "/health", None).await;
    assert_eq!(status, 200);
    assert_eq!(json(&body)["status"], "ok");
    assert_eq!(json(&body)["models"][0], "memory");

    let (status, _, body) = call(app(MemoryBackend::new("x")), "GET", "/v1/models", None).await;
    assert_eq!(status, 200);
    let v = json(&body);
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "memory");
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][0]["owned_by"], "infy");
    assert_eq!(v["data"][0]["created"], 1_700_000_000u64);
}

#[tokio::test]
async fn chat_completion_not_streamed() {
    let (status, ctype, body) = call(
        app(MemoryBackend::new("Hello world")),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"model":"anything","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(ctype.starts_with("application/json"));
    let v = json(&body);
    assert_eq!(v["id"], "chatcmpl-1");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["created"], 1_700_000_000u64);
    assert_eq!(v["model"], "anything");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "Hello world");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn chat_completion_streamed_as_sse() {
    let (status, ctype, body) = call(
        app(MemoryBackend::new("Hello world")),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert!(ctype.starts_with("text/event-stream"), "{ctype}");
    let frames: Vec<&str> = body.lines().filter(|l| l.starts_with("data:")).collect();
    assert_eq!(frames.len(), 5, "{body}");
    let chunks: Vec<serde_json::Value> = frames
        .iter()
        .filter_map(|l| data_line(l))
        .map(json)
        .collect();
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "Hello ");
    assert_eq!(chunks[2]["choices"][0]["delta"]["content"], "world");
    assert_eq!(chunks[3]["choices"][0]["finish_reason"], "stop");
    assert_eq!(chunks[3]["usage"]["completion_tokens"], 2);
    assert_eq!(chunks[3]["model"], "memory");
    assert!(chunks
        .iter()
        .all(|c| c["id"] == "chatcmpl-1" && c["object"] == "chat.completion.chunk"));
    assert_eq!(frames.last(), Some(&"data: [DONE]"));
}

#[tokio::test]
async fn streamed_errors_arrive_as_a_frame_then_done() {
    let (status, _, body) = call(
        app(MemoryBackend::new("one two three")),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"model":"explode","messages":[{"role":"user","content":"hi"}],"stream":true}"#),
    )
    .await;
    assert_eq!(status, 200);
    let chunks: Vec<serde_json::Value> = body.lines().filter_map(data_line).map(json).collect();
    assert_eq!(chunks[1]["choices"][0]["delta"]["content"], "one ");
    assert!(
        chunks[2]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("fell over"),
        "{body}"
    );
    assert_eq!(chunks[2]["error"]["code"], "unavailable");
    assert!(body.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn bad_requests_get_the_openai_error_envelope() {
    let app_ = || app(MemoryBackend::new("x"));
    let (status, _, body) = call(app_(), "POST", "/v1/chat/completions", Some("{not json")).await;
    assert_eq!(status, 400);
    assert_eq!(json(&body)["error"]["type"], "invalid_request_error");
    assert!(json(&body)["error"]["message"]
        .as_str()
        .unwrap()
        .contains("request body"));

    let (status, _, body) = call(
        app_(),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"hi"}],"temperature":-1}"#),
    )
    .await;
    assert_eq!(status, 400);
    assert!(json(&body)["error"]["message"]
        .as_str()
        .unwrap()
        .contains("temperature"));

    let (status, _, body) = call(
        app_(),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[]}"#),
    )
    .await;
    assert_eq!(status, 400, "{body}");

    let (status, _, body) = call(
        app_(),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"model":"missing","messages":[{"role":"user","content":"hi"}]}"#),
    )
    .await;
    assert_eq!(status, 404, "{body}");
    assert_eq!(json(&body)["error"]["code"], "not_found");

    let (status, _, body) = call(
        app_(),
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"hi"}],"n":3}"#),
    )
    .await;
    assert_eq!(status, 400, "{body}");
}

#[tokio::test]
async fn completions_endpoint_both_ways() {
    let (status, _, body) = call(
        app(MemoryBackend::new("foo bar")),
        "POST",
        "/v1/completions",
        Some(r#"{"model":"m","prompt":"Once","max_tokens":3,"temperature":0}"#),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v = json(&body);
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["id"], "cmpl-1");
    assert_eq!(v["choices"][0]["text"], "foo bar");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    let (status, ctype, body) = call(
        app(MemoryBackend::new("foo bar")),
        "POST",
        "/v1/completions",
        Some(r#"{"prompt":"Once","stream":true}"#),
    )
    .await;
    assert_eq!(status, 200);
    assert!(ctype.starts_with("text/event-stream"));
    let chunks: Vec<serde_json::Value> = body.lines().filter_map(data_line).map(json).collect();
    assert_eq!(chunks.len(), 3, "{body}");
    assert_eq!(chunks[0]["choices"][0]["text"], "foo ");
    assert_eq!(chunks[0]["object"], "text_completion");
    assert_eq!(chunks[2]["choices"][0]["finish_reason"], "stop");
    assert!(body.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test]
async fn parameters_reach_the_backend() {
    let backend = Arc::new(MemoryBackend::new("x"));
    let server = Server::new(
        backend.clone(),
        Arc::new(MemoryIds::default()),
        Arc::new(MemoryClock(1)),
        SamplingParams::default(),
    );
    let app = router(Arc::new(server));
    call(app, "POST", "/v1/chat/completions", Some(r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":"u"}],"max_tokens":9,"seed":4,"stop":["x"]}"#)).await;
    let calls = backend.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].model.as_deref(), Some("m"));
    assert_eq!(calls[0].prompt, "system: s\nuser: u");
    assert_eq!(calls[0].params.max_tokens, 9);
    assert_eq!(calls[0].params.seed, Some(4));
    assert_eq!(calls[0].params.stop, vec!["x".to_string()]);
}

#[tokio::test]
async fn dropping_the_response_stream_cancels_the_generation() {
    let (tx, rx) = mpsc::channel::<Frame>(4);
    let cancel = Cancel::new();
    let stream = CancelOnDrop {
        inner: ReceiverStream::new(rx),
        cancel: cancel.clone(),
    };
    assert!(!cancel.is_cancelled());
    drop(stream);
    assert!(cancel.is_cancelled());
    drop(tx);

    // And end to end: a backend that hangs until cancelled is released when
    // the client stops reading.
    let mut backend = MemoryBackend::new("never");
    backend.hang_until_cancelled = true;
    let backend = Arc::new(backend);
    let server = Server::new(
        backend.clone(),
        Arc::new(MemoryIds::default()),
        Arc::new(MemoryClock(1)),
        SamplingParams::default(),
    );
    let app = router(Arc::new(server));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    drop(resp);
    // The worker observes the cancel and finishes; give it a moment.
    for _ in 0..100 {
        if backend.calls().len() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert_eq!(backend.calls().len(), 1);
}
