//! The imperative shell: the models directory on disk, and a downloader.

use std::path::{Path, PathBuf};

use infy_kernel::{Error, ModelRef, Result};

use crate::deps::Downloader;
use crate::logic::{choose_gguf, local_dir_name, parse_ref, GgufSelector, Source};

/// A GGUF in the models directory (or pointed at by a path ref).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModel {
    /// The path relative to the models directory, e.g.
    /// `Qwen--Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf`,
    /// or the absolute path for a file outside it.
    pub name: String,
    pub path: PathBuf,
    pub size_bytes: u64,
}

/// The catalogue and the downloader.
pub struct Hub {
    downloader: Box<dyn Downloader>,
    models_dir: PathBuf,
}

impl Hub {
    pub fn new(downloader: Box<dyn Downloader>, models_dir: impl Into<PathBuf>) -> Self {
        Self {
            downloader,
            models_dir: models_dir.into(),
        }
    }

    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// Every GGUF under the models directory, two levels deep, sorted by name.
    pub fn list(&self) -> Result<Vec<LocalModel>> {
        let mut out = Vec::new();
        if !self.models_dir.exists() {
            return Ok(out);
        }
        for entry in read_dir(&self.models_dir)? {
            if entry.is_dir() {
                for inner in read_dir(&entry)? {
                    self.push_if_gguf(&inner, &mut out)?;
                }
            } else {
                self.push_if_gguf(&entry, &mut out)?;
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    fn push_if_gguf(&self, path: &Path, out: &mut Vec<LocalModel>) -> Result<()> {
        if path.extension().and_then(|e| e.to_str()) == Some("gguf") && path.is_file() {
            out.push(self.local_model(path)?);
        }
        Ok(())
    }

    fn local_model(&self, path: &Path) -> Result<LocalModel> {
        let size_bytes = std::fs::metadata(path)
            .map_err(|e| Error::wrap(format!("reading {}", path.display()), e))?
            .len();
        let name = path
            .strip_prefix(&self.models_dir)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| path.to_string_lossy().into_owned());
        Ok(LocalModel {
            name,
            path: path.to_path_buf(),
            size_bytes,
        })
    }

    /// The local file a ref names, if it is already here. Never touches the
    /// network.
    pub fn find(&self, r: &ModelRef) -> Result<Option<LocalModel>> {
        match parse_ref(r)? {
            Source::Path(p) => {
                let p = expand_home(&p);
                if p.is_file() {
                    Ok(Some(self.local_model(&p)?))
                } else {
                    Ok(None)
                }
            }
            Source::Local(name) => {
                for candidate in [
                    self.models_dir.join(&name),
                    self.models_dir.join(format!("{name}.gguf")),
                ] {
                    if candidate.is_file() {
                        return Ok(Some(self.local_model(&candidate)?));
                    }
                }
                // A bare file name that is unique in the catalogue.
                let all = self.list()?;
                let hits: Vec<&LocalModel> = all
                    .iter()
                    .filter(|m| {
                        let file = m.path.file_name().and_then(|f| f.to_str()).unwrap_or("");
                        file == name || file.strip_suffix(".gguf") == Some(name.as_str())
                    })
                    .collect();
                match hits.as_slice() {
                    [one] => Ok(Some((*one).clone())),
                    [] => Ok(None),
                    many => Err(Error::invalid(format!(
                        "{name:?} matches {} models: {}; use the full name",
                        many.len(),
                        many.iter()
                            .map(|m| m.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))),
                }
            }
            Source::Hf { repo, select, .. } => {
                let dir = self.models_dir.join(local_dir_name(&repo));
                if !dir.is_dir() {
                    return Ok(None);
                }
                let files: Vec<String> = read_dir(&dir)?
                    .iter()
                    .filter(|p| p.is_file())
                    .filter_map(|p| p.file_name().and_then(|f| f.to_str()).map(String::from))
                    .collect();
                // The same selection rule as for the remote listing, so a
                // repository pulled once is found by the same ref afterwards.
                match choose_gguf(&files, &select) {
                    Ok(f) => Ok(Some(self.local_model(&dir.join(f))?)),
                    Err(_) => Ok(None),
                }
            }
            Source::Runtime { runtime, model } => Err(Error::invalid(format!(
                "{r} is served by {runtime}, not a file; ask the runtime for {model:?}"
            ))),
        }
    }

    /// Download what a ref names into the models directory. A file already
    /// present with the expected size is not fetched again.
    pub fn pull(
        &self,
        r: &ModelRef,
        progress: &mut dyn FnMut(&str, u64, u64),
    ) -> Result<LocalModel> {
        let (repo, revision, select) = match parse_ref(r)? {
            Source::Hf {
                repo,
                revision,
                select,
            } => (repo, revision, select),
            Source::Runtime { runtime, .. } => {
                return Err(Error::invalid(format!(
                    "{r} is a {runtime} model; pull it with that runtime (`infy runtime` knows how)"
                )))
            }
            Source::Path(_) | Source::Local(_) => {
                return match self.find(r)? {
                    Some(m) => Ok(m),
                    None => Err(Error::not_found(format!(
                        "{r}: no such file; a local ref cannot be downloaded, use hf:owner/repo"
                    ))),
                };
            }
        };
        let listing = self.downloader.list_files(&repo, revision.as_deref())?;
        let names: Vec<String> = listing.iter().map(|f| f.name.clone()).collect();
        let file = choose_gguf(&names, &select)?;
        let expected = listing.iter().find(|f| f.name == file).and_then(|f| f.size);
        let dir = self.models_dir.join(local_dir_name(&repo));
        let dest = dir.join(&file);
        if let (Ok(meta), Some(size)) = (std::fs::metadata(&dest), expected) {
            if meta.is_file() && meta.len() == size {
                progress(&file, size, size);
                return self.local_model(&dest);
            }
        }
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::wrap(format!("creating {}", dir.display()), e))?;
        let mut on_bytes = |done: u64, total: u64| progress(&file, done, total);
        let path =
            self.downloader
                .download(&repo, revision.as_deref(), &file, &dir, &mut on_bytes)?;
        let path = if path.is_absolute() {
            path
        } else {
            dir.join(path)
        };
        self.local_model(&path)
    }

    /// The file a ref names, pulling it if it is not here yet.
    pub fn resolve(
        &self,
        r: &ModelRef,
        progress: &mut dyn FnMut(&str, u64, u64),
    ) -> Result<LocalModel> {
        if let Some(m) = self.find(r)? {
            return Ok(m);
        }
        match parse_ref(r)? {
            Source::Hf { .. } => self.pull(r, progress),
            Source::Path(p) => Err(Error::not_found(format!(
                "model file {}",
                expand_home(&p).display()
            ))),
            Source::Local(name) => Err(Error::not_found(format!(
                "{name:?} in {}; see `infy models`, or pull one with `infy pull hf:owner/repo`",
                self.models_dir.display()
            ))),
            Source::Runtime { .. } => Err(Error::invalid(format!(
                "{r} is a runtime model, not a file"
            ))),
        }
    }

    pub fn selector_is_file(select: &GgufSelector) -> bool {
        matches!(select, GgufSelector::File(_))
    }
}

fn read_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let rd =
        std::fs::read_dir(dir).map_err(|e| Error::wrap(format!("listing {}", dir.display()), e))?;
    let mut out = Vec::new();
    for entry in rd {
        let entry = entry.map_err(|e| Error::wrap(format!("listing {}", dir.display()), e))?;
        out.push(entry.path());
    }
    out.sort();
    Ok(out)
}

/// `~/x` → `$HOME/x`, when a home directory is known. Done here rather than
/// by the shell because the ref may arrive over an API.
fn expand_home(p: &Path) -> PathBuf {
    if let Ok(rest) = p.strip_prefix("~") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    p.to_path_buf()
}

#[cfg(test)]
#[path = "service_test.rs"]
mod tests;
