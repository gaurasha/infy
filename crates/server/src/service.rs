//! The imperative shell: routes, handlers, and the bridge from a blocking
//! backend to an async response.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;

use infy_kernel::{Cancel, Completion, Error, Result, SamplingParams};
use infy_wire::{
    chat_request_to_kernel, chat_response, completion_chunk, completion_request_to_kernel,
    completion_response, ChatCompletionRequest, CompletionRequest, ModelList, ModelObject,
};

use crate::deps::{Backend, Clock, IdGen};
use crate::logic::{delta_chunk, error_status, final_chunk, first_chunk, requested_model};

/// The server's dependencies and defaults.
pub struct Server {
    backend: Arc<dyn Backend>,
    ids: Arc<dyn IdGen>,
    clock: Arc<dyn Clock>,
    /// Sampling parameters for anything a request leaves unset.
    defaults: SamplingParams,
}

impl Server {
    pub fn new(
        backend: Arc<dyn Backend>,
        ids: Arc<dyn IdGen>,
        clock: Arc<dyn Clock>,
        defaults: SamplingParams,
    ) -> Self {
        Self {
            backend,
            ids,
            clock,
            defaults,
        }
    }
}

/// The router, ready to serve.
pub fn router(server: Arc<Server>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .with_state(server)
}

/// Bind and serve until Ctrl-C.
pub async fn serve(addr: &str, app: Router) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| Error::wrap(format!("listening on {addr}"), e))?;
    tracing::info!(url = %format!("http://{addr}/v1"), "infy is listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .map_err(|e| Error::wrap("serving", e))
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    models: Vec<String>,
}

async fn health(State(s): State<Arc<Server>>) -> Json<Health> {
    Json(Health {
        status: "ok",
        models: s.backend.models().into_iter().map(|m| m.id).collect(),
    })
}

async fn models(State(s): State<Arc<Server>>) -> Json<ModelList> {
    let created = s.clock.unix_seconds();
    Json(ModelList::new(
        s.backend
            .models()
            .into_iter()
            .map(|m| ModelObject::new(m.id, created, m.owned_by))
            .collect(),
    ))
}

fn error(err: &Error) -> Response {
    let (status, body) = error_status(err);
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (status, Json(body)).into_response()
}

fn rejection(r: JsonRejection) -> Response {
    error(&Error::invalid(format!("request body: {}", r.body_text())))
}

/// One streamed item: a JSON payload, or the terminal marker.
enum Frame {
    Json(String),
    Done,
}

fn json<T: Serialize>(v: &T) -> Frame {
    match serde_json::to_string(v) {
        Ok(s) => Frame::Json(s),
        Err(e) => Frame::Json(format!(
            "{{\"error\":{{\"message\":\"encoding a frame: {e}\"}}}}"
        )),
    }
}

/// How one endpoint frames a stream: an optional opening frame, a frame
/// per text delta, and the closing frame.
struct Framing {
    first: Option<fn(&str, u64, &str) -> Frame>,
    delta: fn(&str, u64, &str, &str) -> Frame,
    last: fn(&str, u64, &Completion) -> Frame,
}

/// Runs a blocking generation on a worker thread, forwarding frames into a
/// server-sent-events response.
fn spawn_stream<F>(
    s: &Server,
    prefix: &'static str,
    model_hint: String,
    run: F,
    framing: Framing,
) -> Response
where
    F: FnOnce(&Cancel, &mut dyn FnMut(&str)) -> Result<Completion> + Send + 'static,
{
    let (ids, clock) = (s.ids.clone(), s.clock.clone());
    let Framing {
        first: make_first,
        delta: make_delta,
        last: make_final,
    } = framing;
    let (tx, rx) = mpsc::channel::<Frame>(64);
    let cancel = Cancel::new();
    let worker_cancel = cancel.clone();
    let id = ids.new_id(prefix);
    let created = clock.unix_seconds();
    tokio::task::spawn_blocking(move || {
        if let Some(first) = make_first {
            let _ = tx.blocking_send(first(&id, created, &model_hint));
        }
        let sink_tx = tx.clone();
        let sink_cancel = worker_cancel.clone();
        let (sid, smodel) = (id.clone(), model_hint.clone());
        let mut sink = move |text: &str| {
            if sink_tx
                .blocking_send(make_delta(&sid, created, &smodel, text))
                .is_err()
            {
                // The client is gone; stop computing for nobody.
                sink_cancel.cancel();
            }
        };
        let result = run(&worker_cancel, &mut sink);
        match result {
            Ok(c) => {
                let _ = tx.blocking_send(make_final(&id, created, &c));
            }
            Err(e) => {
                let (_, body) = error_status(&e);
                let _ = tx.blocking_send(json(&body));
            }
        }
        let _ = tx.blocking_send(Frame::Done);
    });
    let stream = CancelOnDrop {
        inner: ReceiverStream::new(rx),
        cancel,
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Cancels the generation when the response stream is dropped, which is
/// what happens when the client disconnects.
struct CancelOnDrop<S> {
    inner: S,
    cancel: Cancel,
}

impl<S: Stream<Item = Frame> + Unpin> Stream for CancelOnDrop<S> {
    type Item = std::result::Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Frame::Json(s))) => Poll::Ready(Some(Ok(Event::default().data(s)))),
            Poll::Ready(Some(Frame::Done)) => {
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

async fn chat(
    State(s): State<Arc<Server>>,
    payload: std::result::Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(p) => p,
        Err(r) => return rejection(r),
    };
    let (messages, params) = match chat_request_to_kernel(&req, &s.defaults) {
        Ok(x) => x,
        Err(e) => return error(&e),
    };
    let model = requested_model(req.model.as_deref()).map(String::from);
    let backend = s.backend.clone();
    if req.stream.unwrap_or(false) {
        let hint = model.clone().unwrap_or_default();
        return spawn_stream(
            &s,
            "chatcmpl",
            hint,
            move |cancel, sink| backend.chat(cancel, model.as_deref(), &messages, &params, sink),
            Framing {
                first: Some(|id, created, model| json(&first_chunk(id, created, model))),
                delta: |id, created, model, text| json(&delta_chunk(id, created, model, text)),
                last: |id, created, c| json(&final_chunk(id, created, &c.model, c.finish, c.usage)),
            },
        );
    }
    let cancel = Cancel::new();
    let result = tokio::task::spawn_blocking(move || {
        backend.chat(&cancel, model.as_deref(), &messages, &params, &mut |_| {})
    })
    .await
    .unwrap_or_else(|e| Err(Error::internal(format!("generation thread failed: {e}"))));
    match result {
        Ok(c) => Json(chat_response(
            &s.ids.new_id("chatcmpl"),
            s.clock.unix_seconds(),
            &c,
        ))
        .into_response(),
        Err(e) => error(&e),
    }
}

async fn completions(
    State(s): State<Arc<Server>>,
    payload: std::result::Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(p) => p,
        Err(r) => return rejection(r),
    };
    let (prompt, params) = match completion_request_to_kernel(&req, &s.defaults) {
        Ok(x) => x,
        Err(e) => return error(&e),
    };
    let model = requested_model(req.model.as_deref()).map(String::from);
    let backend = s.backend.clone();
    if req.stream.unwrap_or(false) {
        let hint = model.clone().unwrap_or_default();
        return spawn_stream(
            s.ids.clone(),
            s.clock.clone(),
            "cmpl",
            hint,
            move |cancel, sink| backend.complete(cancel, model.as_deref(), &prompt, &params, sink),
            None,
            |id, created, model, text| {
                json(&completion_chunk(id, created, model, text, None, None))
            },
            |id, created, c| {
                json(&completion_chunk(
                    id,
                    created,
                    &c.model,
                    "",
                    Some(c.finish),
                    Some(c.usage),
                ))
            },
        );
    }
    let cancel = Cancel::new();
    let result = tokio::task::spawn_blocking(move || {
        backend.complete(&cancel, model.as_deref(), &prompt, &params, &mut |_| {})
    })
    .await
    .unwrap_or_else(|e| Err(Error::internal(format!("generation thread failed: {e}"))));
    match result {
        Ok(c) => Json(completion_response(
            &s.ids.new_id("cmpl"),
            s.clock.unix_seconds(),
            &c,
        ))
        .into_response(),
        Err(e) => error(&e),
    }
}

#[cfg(test)]
#[path = "service_test.rs"]
mod tests;
