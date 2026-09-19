//! An in-memory repository store, shipped with the domain so its tests need
//! no network. Files are byte vectors; downloading writes them to disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use infy_kernel::{Error, Result};

use crate::deps::{Downloader, RemoteFile};

/// A repository's files: name and contents.
type Files = Vec<(String, Vec<u8>)>;

#[derive(Default)]
pub struct MemoryDownloader {
    repos: Mutex<HashMap<String, Files>>,
    downloads: Mutex<Vec<(String, String)>>,
}

impl MemoryDownloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a file to a repository.
    pub fn add(self, repo: &str, file: &str, bytes: &[u8]) -> Self {
        if let Ok(mut r) = self.repos.lock() {
            r.entry(repo.to_string())
                .or_default()
                .push((file.to_string(), bytes.to_vec()));
        }
        self
    }

    /// Every `(repo, file)` actually downloaded, in order.
    pub fn downloads(&self) -> Vec<(String, String)> {
        self.downloads.lock().map(|d| d.clone()).unwrap_or_default()
    }
}

impl Downloader for MemoryDownloader {
    fn list_files(&self, repo: &str, _revision: Option<&str>) -> Result<Vec<RemoteFile>> {
        let repos = self
            .repos
            .lock()
            .map_err(|_| Error::internal("memory downloader poisoned"))?;
        match repos.get(repo) {
            Some(files) => Ok(files
                .iter()
                .map(|(n, b)| RemoteFile {
                    name: n.clone(),
                    size: Some(b.len() as u64),
                })
                .collect()),
            None => Err(Error::not_found(format!("repository {repo}"))),
        }
    }

    fn download(
        &self,
        repo: &str,
        _revision: Option<&str>,
        file: &str,
        dest_dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<PathBuf> {
        let bytes = {
            let repos = self
                .repos
                .lock()
                .map_err(|_| Error::internal("memory downloader poisoned"))?;
            repos
                .get(repo)
                .and_then(|files| files.iter().find(|(n, _)| n == file))
                .map(|(_, b)| b.clone())
                .ok_or_else(|| Error::not_found(format!("{file} in {repo}")))?
        };
        let total = bytes.len() as u64;
        progress(0, total);
        let dest = dest_dir.join(file);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::wrap(format!("creating {}", parent.display()), e))?;
        }
        std::fs::write(&dest, &bytes)
            .map_err(|e| Error::wrap(format!("writing {}", dest.display()), e))?;
        progress(total, total);
        if let Ok(mut d) = self.downloads.lock() {
            d.push((repo.to_string(), file.to_string()));
        }
        Ok(dest)
    }
}
