//! The Hugging Face hub, through the `hf-hub` client.
//!
//! The token is passed in explicitly: the composition root reads `HF_TOKEN`
//! once and removes it from the environment, so nothing here may look for
//! it there.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use hf_hub::progress::{DownloadEvent, Progress, ProgressEvent, ProgressHandler};
use hf_hub::HFClientSync;
use hf_hub::{split_id, HFClientBuilder, HFError};

use infy_kernel::{Error, Result};

use crate::deps::{Downloader, RemoteFile};

pub struct HfDownloader {
    client: HFClientSync,
}

impl HfDownloader {
    /// `token` unlocks gated repositories; `cache_dir` is where the client
    /// keeps its own bookkeeping (not where models land -- that is the
    /// models directory, passed per download).
    pub fn new(token: Option<String>, cache_dir: Option<PathBuf>) -> Result<Self> {
        let mut b = HFClientBuilder::new().user_agent("infy");
        if let Some(t) = token {
            b = b.token(t);
        }
        if let Some(c) = cache_dir {
            b = b.cache_dir(c);
        }
        let client = b
            .build_sync()
            .map_err(|e| Error::wrap("creating the hub client", e))?;
        Ok(Self { client })
    }
}

fn map_err(context: &str, e: HFError) -> Error {
    let text = e.to_string();
    if text.contains("404") || text.to_ascii_lowercase().contains("not found") {
        Error::not_found(format!("{context}: {text}"))
    } else if text.contains("401") || text.contains("403") {
        Error::unavailable(format!(
            "{context}: {text}. If the repository is gated, accept its terms on huggingface.co and set HF_TOKEN"
        ))
    } else {
        Error::unavailable(format!("{context}: {text}"))
    }
}

struct ChannelProgress {
    file: String,
    tx: mpsc::Sender<(u64, u64)>,
}

impl ProgressHandler for ChannelProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        if let ProgressEvent::Download(DownloadEvent::Progress { files }) = event {
            for f in files {
                if f.filename == self.file {
                    let _ = self.tx.send((f.bytes_completed, f.total_bytes));
                }
            }
        }
    }
}

impl Downloader for HfDownloader {
    fn list_files(&self, repo: &str, revision: Option<&str>) -> Result<Vec<RemoteFile>> {
        let (owner, name) = split_id(repo);
        let info = self
            .client
            .model(owner, name)
            .info()
            .maybe_revision(revision.map(String::from))
            .send()
            .map_err(|e| map_err(&format!("listing {repo}"), e))?;
        Ok(info
            .siblings
            .unwrap_or_default()
            .into_iter()
            .map(|s| RemoteFile {
                name: s.rfilename,
                size: s.size,
            })
            .collect())
    }

    fn download(
        &self,
        repo: &str,
        revision: Option<&str>,
        file: &str,
        dest_dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<PathBuf> {
        let (owner, name) = split_id(repo);
        let repository = self.client.model(owner, name);
        let (tx, rx) = mpsc::channel();
        let handler = Progress::new(ChannelProgress {
            file: file.to_string(),
            tx,
        });
        let dest_dir = dest_dir.to_path_buf();
        let revision = revision.map(String::from);
        let file_name = file.to_string();
        // The client blocks for the whole download and reports progress from
        // inside; draining the channel on this thread while it runs on
        // another is what lets a borrowed callback see the updates.
        std::thread::scope(|s| {
            let worker = s.spawn(move || {
                repository
                    .download_file()
                    .filename(file_name)
                    .local_dir(dest_dir)
                    .maybe_revision(revision)
                    .progress(handler)
                    .send()
            });
            for (done, total) in rx {
                progress(done, total);
            }
            match worker.join() {
                Ok(r) => r.map_err(|e| map_err(&format!("downloading {file} from {repo}"), e)),
                Err(_) => Err(Error::internal(format!("download of {file} panicked"))),
            }
        })
    }
}
