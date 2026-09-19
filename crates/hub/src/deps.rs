use std::path::{Path, PathBuf};

use infy_kernel::Result;

/// One file in a remote repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    pub name: String,
    pub size: Option<u64>,
}

/// Something that can list and fetch a repository's files. Implementations
/// live in `deps/`: the Hugging Face hub, and an in-memory one that ships
/// with the domain so its tests need no network.
pub trait Downloader: Send + Sync {
    fn list_files(&self, repo: &str, revision: Option<&str>) -> Result<Vec<RemoteFile>>;

    /// Fetch `file` into `dest_dir`, calling `progress` with bytes done and
    /// bytes total as it goes, and return the file's path.
    fn download(
        &self,
        repo: &str,
        revision: Option<&str>,
        file: &str,
        dest_dir: &Path,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<PathBuf>;
}

#[cfg(feature = "hf")]
pub mod hfhub;
pub mod memory;
