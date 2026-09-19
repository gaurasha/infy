use serde::{Deserialize, Serialize};

use infy_kernel::Error;

/// A machine-readable error class. Clients key off this rather than parsing
/// prose, so wording can change without breaking them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Code {
    BadRequest,
    NotFound,
    Unavailable,
    Cancelled,
    Internal,
}

impl Code {
    pub fn as_str(self) -> &'static str {
        match self {
            Code::BadRequest => "bad_request",
            Code::NotFound => "not_found",
            Code::Unavailable => "unavailable",
            Code::Cancelled => "cancelled",
            Code::Internal => "internal",
        }
    }

    /// The OpenAI `type` field, which their SDKs switch on.
    pub fn openai_type(self) -> &'static str {
        match self {
            Code::BadRequest => "invalid_request_error",
            Code::NotFound => "not_found_error",
            Code::Unavailable => "service_unavailable_error",
            Code::Cancelled => "cancelled_error",
            Code::Internal => "server_error",
        }
    }

    pub fn http_status(self) -> u16 {
        match self {
            Code::BadRequest => 400,
            Code::NotFound => 404,
            Code::Unavailable => 503,
            // nginx's "client closed request": nobody is listening anyway.
            Code::Cancelled => 499,
            Code::Internal => 500,
        }
    }
}

/// Maps a kernel sentinel to its protocol code, so transports do not each
/// re-derive the mapping and drift apart.
pub fn code_for(err: &Error) -> Code {
    match err {
        Error::Invalid(_) => Code::BadRequest,
        Error::NotFound(_) => Code::NotFound,
        Error::Unavailable(_) => Code::Unavailable,
        Error::Cancelled => Code::Cancelled,
        Error::Internal { .. } => Code::Internal,
    }
}

/// The OpenAI error envelope: `{"error": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

/// Build the envelope for an error. The message is the error's own text,
/// which by the failure discipline in GUIDELINES.md already says what is
/// wrong and what to do.
pub fn error_response(err: &Error) -> ErrorResponse {
    let code = code_for(err);
    ErrorResponse {
        error: ErrorBody {
            message: err.to_string(),
            kind: code.openai_type().to_string(),
            code: Some(code.as_str().to_string()),
            param: None,
        },
    }
}
