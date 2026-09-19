//! Server-sent-events framing, as the OpenAI protocol uses it: every frame is
//! `data: <json>\n\n`, and the stream ends with `data: [DONE]\n\n`.

/// The terminal frame.
pub const DONE_FRAME: &str = "data: [DONE]\n\n";

/// Wrap one JSON payload as an SSE frame.
pub fn frame(json: &str) -> String {
    format!("data: {json}\n\n")
}

/// Decode one line of an SSE stream from the client side.
///
/// Returns `Some(payload)` for a data line that is not the terminal marker,
/// and `None` for the terminal marker, comments, blank lines and other
/// fields. Callers accumulate nothing: every runtime we talk to sends one
/// JSON object per data line.
pub fn data_line(line: &str) -> Option<&str> {
    let payload = line.strip_prefix("data:")?.trim();
    if payload.is_empty() || payload == "[DONE]" {
        None
    } else {
        Some(payload)
    }
}
