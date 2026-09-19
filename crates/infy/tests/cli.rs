//! End to end through the real binary: a tiny GGUF is written to a temp
//! models directory, then `infy models`, `infy generate` and `infy serve`
//! run against it. Random weights produce random text; what this proves is
//! that the whole path holds together -- file, loader, architecture,
//! tokenizer rebuilt from the file's vocabulary, chat template, engine,
//! CLI and HTTP server.

use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use infy_kernel::Vocabulary;
use infy_models::deps::gguf::write_gguf;
use infy_models::deps::memory::RandomModel;
use infy_models::{Architecture, ModelConfig, RopeScaling};
use infy_runtime::deps::system::UreqHttp;
use infy_runtime::Http;

const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

/// GPT-2's byte → printable-character mapping, the alphabet of every
/// byte-level BPE vocabulary.
fn byte_alphabet() -> Vec<String> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<char> = bytes.iter().map(|&b| b as char).collect();
    let mut n = 0u32;
    for b in 0..=255u8 {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    let mut pairs: Vec<(u8, char)> = bytes.into_iter().zip(chars).collect();
    pairs.sort_by_key(|(b, _)| *b);
    pairs.into_iter().map(|(_, c)| c.to_string()).collect()
}

fn write_tiny_model(dir: &Path) {
    let mut tokens = byte_alphabet();
    let merges = ["h i", "Ġ t", "Ġt h"];
    for m in merges {
        tokens.push(m.replace(' ', ""));
    }
    let mut types = vec![1i32; tokens.len()];
    for special in ["<|im_start|>", "<|im_end|>"] {
        tokens.push(special.into());
        types.push(3);
    }
    let eos = tokens.len() as u32 - 1;
    let vocab = Vocabulary {
        model: "gpt2".into(),
        pre: Some("default".into()),
        tokens: tokens.clone(),
        token_types: types,
        merges: merges.iter().map(|m| m.to_string()).collect(),
        eos_token_ids: vec![eos],
        add_bos_token: Some(false),
        chat_template: Some(CHATML.into()),
        ..Default::default()
    };
    let config = ModelConfig {
        architecture: Architecture::Llama,
        name: "tiny".into(),
        hidden_size: 32,
        intermediate_size: 64,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 8,
        vocab_size: tokens.len(),
        context_length: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        rope_scaling: RopeScaling::None,
    };
    let src = RandomModel::new(&config, 5, candle_dtype()).with_vocabulary(&vocab);
    std::fs::create_dir_all(dir).unwrap();
    write_gguf(dir.join("tiny.gguf"), &src).unwrap();
}

fn candle_dtype() -> infy_models::GgmlDType {
    infy_models::GgmlDType::Q8_0
}

fn infy(data_dir: &Path, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_infy"))
        .arg("--data-dir")
        .arg(data_dir)
        .arg("--device")
        .arg("cpu")
        .args(args)
        .env_remove("INFY_MODEL")
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn models_generate_and_serve_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let data = tmp.path();
    write_tiny_model(&data.join("models"));

    // --- models ---
    let (code, out, err) = infy(data, &["models"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("tiny.gguf"), "{out}");
    assert!(out.contains("ollama"), "{out}");

    // --- generate, through the chat template ---
    let (code, out, err) = infy(
        data,
        &[
            "generate",
            "--max-tokens",
            "6",
            "--temperature",
            "0",
            "--seed",
            "1",
            "--show-prompt",
            "hi there",
        ],
    );
    assert_eq!(code, 0, "stdout: {out}\nstderr: {err}");
    assert!(
        err.contains("<|im_start|>user\nhi there<|im_end|>"),
        "the chat template was applied: {err}"
    );
    assert!(err.contains("[tiny.gguf · in "), "stats line: {err}");
    assert!(err.contains("loaded in"), "{err}");

    // --- generate, raw, with the only model picked by default ---
    let (code, _, err) = infy(
        data,
        &[
            "generate",
            "--raw",
            "--max-tokens",
            "3",
            "--seed",
            "2",
            "once upon",
        ],
    );
    assert_eq!(code, 0, "{err}");
    assert!(err.contains("· out 3 ·") || err.contains("Stop]"), "{err}");

    // --- a missing model is a specific error with a non-zero code ---
    let (code, _, err) = infy(data, &["generate", "--model", "nothere", "x"]);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("infy pull"), "{err}");

    // --- serve ---
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let listen = format!("127.0.0.1:{port}");
    let mut child = Command::new(env!("CARGO_BIN_EXE_infy"))
        .args(["--data-dir"])
        .arg(data)
        .args([
            "--device",
            "cpu",
            "serve",
            "--model",
            "tiny.gguf",
            "--listen",
            &listen,
            "--max-tokens",
            "4",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let http = UreqHttp::new();
    let base = format!("http://{listen}");
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if http
            .get(&format!("{base}/health"), Duration::from_secs(1))
            .is_ok()
        {
            break;
        }
        assert!(Instant::now() < deadline, "server did not come up");
        std::thread::sleep(Duration::from_millis(100));
    }
    let models = http
        .get(&format!("{base}/v1/models"), Duration::from_secs(5))
        .unwrap();
    assert_eq!(models.status, 200);
    let v: serde_json::Value = serde_json::from_str(&models.body).unwrap();
    assert_eq!(v["data"][0]["id"], "tiny.gguf");

    let chat = http
        .post_json(
            &format!("{base}/v1/chat/completions"),
            r#"{"model":"tiny.gguf","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0}"#,
            Duration::from_secs(30),
        )
        .unwrap();
    assert_eq!(chat.status, 200, "{}", chat.body);
    let v: serde_json::Value = serde_json::from_str(&chat.body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "tiny.gguf");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);

    let missing = http
        .post_json(
            &format!("{base}/v1/chat/completions"),
            r#"{"model":"ollama:nothere","messages":[{"role":"user","content":"hi"}]}"#,
            Duration::from_secs(10),
        )
        .unwrap();
    assert_eq!(missing.status, 503, "{}", missing.body);

    let _ = child.kill();
    let _ = child.wait();
}
