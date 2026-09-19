use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::{Error, Result};

/// A cancellation flag, cloned into every long-running operation.
///
/// It is the first argument of every function that runs longer than a few
/// milliseconds, and it is honoured between tokens. This is a system built out
/// of long-running generations; an uncancellable one is a hang waiting to
/// happen. An atomic rather than a channel so that it needs nothing beyond
/// `std`, and `Clone` so a server can keep one half and hand the other to the
/// engine.
#[derive(Debug, Clone, Default)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// `Err(Error::Cancelled)` once cancelled, so a loop can write `cancel.check()?`.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}
