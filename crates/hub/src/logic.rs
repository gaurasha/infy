//! Pure logic: what a model ref means, and which file to take from a
//! repository.

use std::path::PathBuf;

use infy_kernel::{Error, ModelRef, Result};

/// Which GGUF to take from a repository.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufSelector {
    /// The repository's obvious choice: `Q4_K_M` if there is one, otherwise
    /// the next preferred quantisation, otherwise the only file.
    Default,
    /// A file whose name contains this quantisation tag, e.g. `Q8_0`.
    Quant(String),
    /// An exact path inside the repository.
    File(String),
}

/// What a model ref names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// A Hugging Face repository.
    Hf {
        repo: String,
        revision: Option<String>,
        select: GgufSelector,
    },
    /// A GGUF file on disk.
    Path(PathBuf),
    /// A file in the models directory, by name.
    Local(String),
    /// A model served by an external runtime.
    Runtime { runtime: String, model: String },
}

/// The runtimes a ref may address. Kept as data here so the hub never has
/// to know what a runtime is, only that the ref is not its business.
pub const RUNTIME_PREFIXES: &[&str] = &["ollama", "llamacpp", "lmstudio"];

/// Parse a ref.
///
/// | form | meaning |
/// |---|---|
/// | `hf:owner/repo` | the default GGUF of that repository |
/// | `hf:owner/repo:Q8_0` | the GGUF whose name contains `Q8_0` |
/// | `hf:owner/repo/sub/file.gguf` | that exact file |
/// | `hf:owner/repo@rev...` | any of the above at a revision |
/// | `owner/repo` | same as `hf:owner/repo` |
/// | `./x.gguf`, `/x.gguf`, `~/x.gguf` | a file on disk |
/// | `ollama:name`, `llamacpp:name`, `lmstudio:name` | a runtime's model |
/// | anything else, `x.gguf` included | a name in the models directory (then the working directory) |
pub fn parse_ref(r: &ModelRef) -> Result<Source> {
    let s = r.as_str();
    if let Some((prefix, rest)) = s.split_once(':') {
        if RUNTIME_PREFIXES.contains(&prefix) {
            if rest.is_empty() {
                return Err(Error::invalid(format!(
                    "{s:?}: no model name after {prefix}:"
                )));
            }
            return Ok(Source::Runtime {
                runtime: prefix.to_string(),
                model: rest.to_string(),
            });
        }
        if prefix == "hf" {
            return parse_hf(rest);
        }
    }
    if s.starts_with('/') || s.starts_with("./") || s.starts_with("../") || s.starts_with("~/") {
        return Ok(Source::Path(PathBuf::from(s)));
    }
    if looks_like_repo(s) {
        return parse_hf(s);
    }
    Ok(Source::Local(s.to_string()))
}

/// `owner/repo`, with exactly one slash and no file extension after it.
fn looks_like_repo(s: &str) -> bool {
    match s.split_once('/') {
        Some((owner, repo)) => {
            !owner.is_empty()
                && !repo.is_empty()
                && !repo.contains('/')
                && !repo.ends_with(".gguf")
                && !owner.starts_with('.')
        }
        None => false,
    }
}

fn parse_hf(rest: &str) -> Result<Source> {
    let (owner, tail) = rest
        .split_once('/')
        .filter(|(o, t)| !o.is_empty() && !t.is_empty())
        .ok_or_else(|| Error::invalid(format!("hf:{rest}: expected hf:owner/repo")))?;
    // owner/repo[@rev][:QUANT] or owner/repo[@rev]/path/file.gguf
    let (repo_part, file) = match tail.split_once('/') {
        Some((r, f)) => (r, Some(f)),
        None => (tail, None),
    };
    let (repo_part, quant) = match file {
        Some(_) => (repo_part, None),
        None => match repo_part.split_once(':') {
            Some((r, q)) if !q.is_empty() => (r, Some(q)),
            Some((_, _)) => {
                return Err(Error::invalid(format!(
                    "hf:{rest}: empty quantisation after ':'"
                )))
            }
            None => (repo_part, None),
        },
    };
    let (repo_name, revision) = match repo_part.split_once('@') {
        Some((r, rev)) if !rev.is_empty() => (r, Some(rev.to_string())),
        Some((_, _)) => {
            return Err(Error::invalid(format!(
                "hf:{rest}: empty revision after '@'"
            )))
        }
        None => (repo_part, None),
    };
    if repo_name.is_empty() {
        return Err(Error::invalid(format!("hf:{rest}: empty repository name")));
    }
    let select = match (file, quant) {
        (Some(f), _) => {
            if !f.ends_with(".gguf") {
                return Err(Error::invalid(format!(
                    "hf:{rest}: {f:?} is not a .gguf file"
                )));
            }
            GgufSelector::File(f.to_string())
        }
        (None, Some(q)) => GgufSelector::Quant(q.to_string()),
        (None, None) => GgufSelector::Default,
    };
    Ok(Source::Hf {
        repo: format!("{owner}/{repo_name}"),
        revision,
        select,
    })
}

/// The directory a repository's files live in under the models directory:
/// `owner--repo`, flat, so `infy models` reads as the refs people typed.
pub fn local_dir_name(repo: &str) -> String {
    repo.replace('/', "--")
}

/// Quantisations in order of preference when a ref does not choose one:
/// the sizes people actually run, best quality-per-byte first.
const PREFERRED_QUANTS: &[&str] = &[
    "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q4_0", "F16", "BF16", "F32",
];

fn is_shard(name: &str) -> bool {
    name.contains("-of-") && name.ends_with(".gguf")
}

fn is_projector(name: &str) -> bool {
    name.to_ascii_lowercase().contains("mmproj")
}

/// Pick one GGUF from a repository's file list.
///
/// Shards (`-00001-of-00003.gguf`) and multimodal projectors (`mmproj-*`)
/// are never candidates. Every error lists what is available, because the
/// fix is always "say which one".
pub fn choose_gguf(files: &[String], select: &GgufSelector) -> Result<String> {
    let ggufs: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    let candidates: Vec<&String> = ggufs
        .iter()
        .copied()
        .filter(|f| !is_shard(f) && !is_projector(f))
        .collect();
    let listing = || {
        if ggufs.is_empty() {
            "the repository has no .gguf files".to_string()
        } else {
            format!(
                "available: {}",
                ggufs
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    };
    match select {
        GgufSelector::File(f) => {
            if files.iter().any(|x| x == f) {
                Ok(f.clone())
            } else {
                Err(Error::not_found(format!(
                    "{f} in the repository; {}",
                    listing()
                )))
            }
        }
        GgufSelector::Quant(q) => {
            let q_upper = q.to_ascii_uppercase();
            let matches: Vec<&String> = candidates
                .iter()
                .copied()
                .filter(|f| f.to_ascii_uppercase().contains(&q_upper))
                .collect();
            match matches.as_slice() {
                [one] => Ok((*one).clone()),
                [] => Err(Error::not_found(format!(
                    "no GGUF matching {q:?}; {}",
                    listing()
                ))),
                many => {
                    // Prefer the exact tag delimited by '-' or '.'.
                    let exact: Vec<&&String> = many
                        .iter()
                        .filter(|f| {
                            let u = f.to_ascii_uppercase();
                            u.contains(&format!("-{q_upper}."))
                                || u.contains(&format!("_{q_upper}."))
                                || u.contains(&format!(".{q_upper}."))
                        })
                        .collect();
                    match exact.as_slice() {
                        [one] => Ok((**one).clone()),
                        _ => Err(Error::invalid(format!(
                            "{q:?} matches several files: {}; name the file with hf:owner/repo/<file>.gguf",
                            many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                        ))),
                    }
                }
            }
        }
        GgufSelector::Default => {
            for q in PREFERRED_QUANTS {
                let hits: Vec<&String> = candidates
                    .iter()
                    .copied()
                    .filter(|f| {
                        f.to_ascii_uppercase().contains(&format!("{q}."))
                            || f.to_ascii_uppercase().contains(&format!("{q}-"))
                    })
                    .collect();
                if let [one] = hits.as_slice() {
                    return Ok((*one).clone());
                }
            }
            match candidates.as_slice() {
                [one] => Ok((*one).clone()),
                [] if !ggufs.is_empty() => Err(Error::invalid(format!(
                    "only sharded or projector GGUFs found, which infy cannot load yet; {}",
                    listing()
                ))),
                [] => Err(Error::not_found(format!("a GGUF to download; {}", listing()))),
                many => Err(Error::invalid(format!(
                    "several GGUFs and none of the usual quantisations; choose with hf:owner/repo:<QUANT>. Available: {}",
                    many.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                ))),
            }
        }
    }
}

#[cfg(test)]
#[path = "logic_test.rs"]
mod tests;
