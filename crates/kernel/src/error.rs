use std::fmt;

/// The result type every domain returns across its boundary.
pub type Result<T> = std::result::Result<T, Error>;

/// The one error type that crosses a domain boundary.
///
/// The variants are sentinels: any domain, and the wire layer, can match on
/// them without importing a sibling. A domain that had to import another
/// domain just to recognise "not found" would violate the dependency rule for
/// no reason.
#[derive(Debug)]
pub enum Error {
    /// An addressed entity does not exist: a model, a file, a runtime.
    NotFound(String),
    /// Input failed validation. A caller mistake, not a system failure; maps
    /// to 4xx at the API boundary.
    Invalid(String),
    /// A dependency this operation needs is not reachable right now: ollama
    /// not running, no GPU, hub unreachable. Explicitly a normal, recoverable
    /// state -- the message says what is wrong and, where there is one, the
    /// command that fixes it.
    Unavailable(String),
    /// The caller asked to stop. Whatever was produced so far is kept.
    Cancelled,
    /// Everything else. `context` locates the failure; `source` keeps the
    /// underlying error reachable for anyone who needs the detail.
    Internal {
        context: String,
        source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    },
}

impl Error {
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::Invalid(msg.into())
    }

    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self::Unavailable(msg.into())
    }

    pub fn internal(context: impl Into<String>) -> Self {
        Self::Internal {
            context: context.into(),
            source: None,
        }
    }

    /// Wrap an underlying error with enough context to locate the failure
    /// without a debugger: `Error::wrap("reading tensor blk.3.attn_q.weight", e)`.
    pub fn wrap(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Internal {
            context: context.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(m) => write!(f, "not found: {m}"),
            Self::Invalid(m) => write!(f, "invalid: {m}"),
            Self::Unavailable(m) => write!(f, "unavailable: {m}"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Internal {
                context,
                source: Some(s),
            } => write!(f, "{context}: {s}"),
            Self::Internal {
                context,
                source: None,
            } => write!(f, "{context}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Internal {
                source: Some(s), ..
            } => Some(s.as_ref()),
            _ => None,
        }
    }
}
