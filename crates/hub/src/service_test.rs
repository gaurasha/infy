use super::*;
use crate::deps::memory::MemoryDownloader;

fn r(s: &str) -> ModelRef {
    ModelRef::new(s).unwrap()
}

fn hub(dir: &Path) -> (Hub, std::sync::Arc<MemoryDownloader>) {
    let d = std::sync::Arc::new(
        MemoryDownloader::new()
            .add("o/r", "README.md", b"hi")
            .add("o/r", "m-Q8_0.gguf", &[8u8; 800])
            .add("o/r", "m-Q4_K_M.gguf", &[4u8; 400])
            .add("o/r", "sub/extra.gguf", &[1u8; 10]),
    );
    struct Shared(std::sync::Arc<MemoryDownloader>);
    impl Downloader for Shared {
        fn list_files(
            &self,
            repo: &str,
            rev: Option<&str>,
        ) -> Result<Vec<crate::deps::RemoteFile>> {
            self.0.list_files(repo, rev)
        }
        fn download(
            &self,
            repo: &str,
            rev: Option<&str>,
            file: &str,
            dest: &Path,
            p: &mut dyn FnMut(u64, u64),
        ) -> Result<PathBuf> {
            self.0.download(repo, rev, file, dest, p)
        }
    }
    (Hub::new(Box::new(Shared(d.clone())), dir), d)
}

#[test]
fn pull_picks_the_default_quant_places_it_under_the_repo_dir_and_reports_progress() {
    let tmp = tempfile::tempdir().unwrap();
    let (hub, dl) = hub(tmp.path());
    let mut events = Vec::new();
    let m = hub
        .pull(&r("hf:o/r"), &mut |f, done, total| {
            events.push((f.to_string(), done, total))
        })
        .unwrap();
    assert_eq!(m.name, "o--r/m-Q4_K_M.gguf");
    assert_eq!(m.path, tmp.path().join("o--r").join("m-Q4_K_M.gguf"));
    assert_eq!(m.size_bytes, 400);
    assert!(m.path.is_file());
    assert_eq!(events.first(), Some(&("m-Q4_K_M.gguf".to_string(), 0, 400)));
    assert_eq!(
        events.last(),
        Some(&("m-Q4_K_M.gguf".to_string(), 400, 400))
    );
    assert_eq!(
        dl.downloads(),
        vec![("o/r".to_string(), "m-Q4_K_M.gguf".to_string())]
    );
}

#[test]
fn pull_is_idempotent_and_selectors_work() {
    let tmp = tempfile::tempdir().unwrap();
    let (hub, dl) = hub(tmp.path());
    hub.pull(&r("hf:o/r"), &mut |_, _, _| {}).unwrap();
    let again = hub.pull(&r("hf:o/r"), &mut |_, _, _| {}).unwrap();
    assert_eq!(again.name, "o--r/m-Q4_K_M.gguf");
    assert_eq!(
        dl.downloads().len(),
        1,
        "a complete file is not fetched again"
    );

    let q8 = hub.pull(&r("hf:o/r:Q8_0"), &mut |_, _, _| {}).unwrap();
    assert_eq!(q8.name, "o--r/m-Q8_0.gguf");
    let sub = hub
        .pull(&r("hf:o/r/sub/extra.gguf"), &mut |_, _, _| {})
        .unwrap();
    assert_eq!(sub.name, "o--r/sub/extra.gguf");
    assert_eq!(dl.downloads().len(), 3);

    // A truncated file is fetched again.
    std::fs::write(&q8.path, b"partial").unwrap();
    let q8 = hub.pull(&r("hf:o/r:Q8_0"), &mut |_, _, _| {}).unwrap();
    assert_eq!(q8.size_bytes, 800);
    assert_eq!(dl.downloads().len(), 4);
}

#[test]
fn list_find_and_resolve() {
    let tmp = tempfile::tempdir().unwrap();
    let (hub, dl) = hub(tmp.path());
    assert!(
        hub.list().unwrap().is_empty(),
        "no directory yet is an empty catalogue"
    );
    assert_eq!(hub.find(&r("hf:o/r")).unwrap(), None);

    let m = hub.resolve(&r("hf:o/r"), &mut |_, _, _| {}).unwrap();
    assert_eq!(dl.downloads().len(), 1);
    assert_eq!(hub.resolve(&r("hf:o/r"), &mut |_, _, _| {}).unwrap(), m);
    assert_eq!(dl.downloads().len(), 1, "resolve finds before it pulls");

    hub.pull(&r("hf:o/r:Q8_0"), &mut |_, _, _| {}).unwrap();
    std::fs::write(tmp.path().join("loose.gguf"), b"xx").unwrap();
    let names: Vec<String> = hub.list().unwrap().into_iter().map(|m| m.name).collect();
    assert_eq!(
        names,
        vec!["loose.gguf", "o--r/m-Q4_K_M.gguf", "o--r/m-Q8_0.gguf"]
    );

    assert_eq!(
        hub.find(&r("hf:o/r")).unwrap().unwrap().name,
        "o--r/m-Q4_K_M.gguf"
    );
    assert_eq!(
        hub.find(&r("hf:o/r:Q8_0")).unwrap().unwrap().name,
        "o--r/m-Q8_0.gguf"
    );
    assert_eq!(
        hub.find(&r("o--r/m-Q8_0.gguf")).unwrap().unwrap().name,
        "o--r/m-Q8_0.gguf"
    );
    assert_eq!(hub.find(&r("loose")).unwrap().unwrap().name, "loose.gguf");
    assert_eq!(
        hub.find(&r("m-Q8_0")).unwrap().unwrap().name,
        "o--r/m-Q8_0.gguf",
        "a unique bare file name"
    );
    assert_eq!(hub.find(&r("nothing")).unwrap(), None);
    assert_eq!(
        hub.find(&r("loose.gguf")).unwrap().unwrap().name,
        "loose.gguf",
        "a bare file name finds the catalogue entry"
    );

    let abs = tmp.path().join("o--r").join("m-Q8_0.gguf");
    let found = hub
        .find(&ModelRef::new(abs.to_string_lossy()).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(found.path, abs);
    assert_eq!(
        found.name, "o--r/m-Q8_0.gguf",
        "inside the models dir the name is relative"
    );
}

#[test]
fn failures_are_specific() {
    let tmp = tempfile::tempdir().unwrap();
    let (hub, _) = hub(tmp.path());
    assert!(
        matches!(hub.pull(&r("hf:o/missing"), &mut |_, _, _| {}), Err(Error::NotFound(m)) if m.contains("o/missing"))
    );
    assert!(
        matches!(hub.pull(&r("ollama:x"), &mut |_, _, _| {}), Err(Error::Invalid(m)) if m.contains("ollama"))
    );
    assert!(matches!(hub.find(&r("ollama:x")), Err(Error::Invalid(_))));
    assert!(
        matches!(hub.pull(&r("./nothere.gguf"), &mut |_, _, _| {}), Err(Error::NotFound(m)) if m.contains("nothere"))
    );
    assert!(
        matches!(hub.resolve(&r("nothere"), &mut |_, _, _| {}), Err(Error::NotFound(m)) if m.contains("infy pull"))
    );
    assert!(
        matches!(hub.resolve(&r("/x/y.gguf"), &mut |_, _, _| {}), Err(Error::NotFound(m)) if m.contains("/x/y.gguf"))
    );
    assert!(
        matches!(hub.pull(&r("hf:o/r:Q2_K"), &mut |_, _, _| {}), Err(Error::NotFound(m)) if m.contains("available"))
    );
    assert_eq!(hub.models_dir(), tmp.path());
    assert!(Hub::selector_is_file(&GgufSelector::File("x".into())));
}
