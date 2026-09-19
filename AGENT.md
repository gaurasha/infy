# AGENT.md

How to navigate and work in infy. Rules for writing code: [GUIDELINES.md](GUIDELINES.md).
Why things are shaped the way they are: [docs/DECISIONS.md](docs/DECISIONS.md).
What the product is meant to do: [docs/product/](docs/product/).

## What this is

A local inference engine: one Rust binary, `infy`, that runs language models on
your own machine and exposes them as a chat REPL and an OpenAI-compatible HTTP
server. It runs models two ways, and the split is the whole architecture:

1. **Natively.** A `.gguf` file is loaded and executed on Rust primitives: our
   own Llama-family architecture and KV cache over candle's quantised kernels.
   This is the path every optimisation in this repository lands on.
2. **Through an external runtime.** ollama, llama.cpp's `llama-server` and LM
   Studio are detected, started, stopped and driven over their OpenAI-compatible
   HTTP APIs. They are data (a `RuntimeSpec`), not code paths.

v1 is text only. Voice, image, video and training come later; the structure is
laid out so they land as new domains, not as rewrites.

## The three things to understand first

**1. No domain depends on a sibling domain.** Each `crates/<domain>/` declares
what it needs as traits in `src/deps.rs`; `crates/infy` supplies the concrete
types through newtype adapters. `infy-kernel` and `infy-wire` are the only
shared leaves. This is enforced by `make check-arch` and `make check-portable`,
so a violation fails the build rather than being noticed in review six months
later.

**2. The boundary between a model and the engine is "tokens in, logits out".**
`engine::Model` never sees a tensor. That is what lets the engine's sampling,
stop handling, streaming and prompt cache be pure, tested by tables, and
identical whether the logits came from candle on a GPU or from the in-memory
test model.

**3. Only bootstrap comes from flags and environment.** Data directory, listen
address, device, log level, `HF_TOKEN`, and runtime overrides. A model exists
because its file is in the models directory. There is no config file, and
adding one would be a regression.

## Layout

```
crates/
  kernel/    TokenId, Message, Usage, SamplingParams, Completion, Vocabulary, Cancel, Error. serde only
  wire/      the OpenAI-compatible HTTP protocol types, error codes, SSE framing. kernel + serde
  models/    model architectures on Rust primitives: GGUF → Llama family, growable KV cache
  engine/    the native generation loop: tokenise, prefill, decode, sample, stream, prompt cache
  runtime/   external engines as data: ollama, llama.cpp, LM Studio -- detect, start, stop, chat
  hub/       where models come from: refs, GGUF selection, download, the local catalogue
  server/    axum: /v1/models, /v1/chat/completions, /v1/completions, /health
  infy/      the composition root and CLI -- the only crate that depends on everything
scripts/     check-arch.sh, check-portable.sh
docs/        DECISIONS.md and product/
```

| Path | What |
|---|---|
| `crates/kernel/` | domain types and sentinel errors. **serde only** |
| `crates/wire/` | the protocol both our server and our client to other runtimes speak |
| `crates/models/` | `Llama` over `QMatMul`, `KvCache`, GGUF metadata → `ModelConfig` + `Vocabulary` |
| `crates/engine/` | `Engine::generate` / `Engine::chat`; sampling and stop logic in `logic.rs` |
| `crates/runtime/` | `RuntimeSpec` table, `Runtimes::{status,start,stop,install,models,chat}` |
| `crates/hub/` | `parse_ref`, `choose_gguf`, `Hub::{pull,resolve,list}` |
| `crates/server/` | the HTTP layer; streams a blocking generation through a channel into SSE |
| `crates/infy/` | `bootstrap.rs` (flags/env, the only place), `adapters.rs` (the seams), `commands/` |
| `docs/DECISIONS.md` | engineering decisions, and the measurements behind them |
| `docs/product/` | what infy is for, what it can do, what is next |

## Commands

`make` with no target prints everything.

| Command | Does |
|---|---|
| `make check` | build, tests, clippy, format, architecture rules -- what CI runs |
| `make test` | `cargo test --workspace` |
| `make check-arch` | fail if a domain depends on a sibling or a leaf grows a dependency |
| `make check-portable` | copy each domain + leaves into an empty workspace and test it there |
| `make chat ARGS="--model hf:..."` | interactive chat |
| `make run ARGS="--model hf:..."` | the OpenAI-compatible server on `127.0.0.1:8321` |
| `make release` | optimised binary at `target/release/infy` |

`cargo test -p infy-engine` and friends work too; the Makefile is there so the
architecture checks are one command away.

The binary:

```
infy chat     [--model REF]          REPL; streams tokens, prints usage and timing per turn
infy generate PROMPT [--model REF]   one completion to stdout
infy serve    [--listen ADDR]        OpenAI-compatible HTTP server
infy pull     REF                    download a GGUF into the models directory
infy models                          local GGUFs, plus what each detected runtime serves
infy runtime  status|start|stop|install|models  ollama|llamacpp|lmstudio
```

A model ref is `hf:owner/repo` (picks a Q4_K_M or the only GGUF),
`hf:owner/repo:Q8_0` (quant tag), `hf:owner/repo/file.gguf`, a local `name`
or `./path.gguf`, or `ollama:`/`llamacpp:`/`lmstudio:` followed by that
runtime's model name.

## Things that will bite you

**Debug builds of tensor code are unusably slow.** `Cargo.toml` sets
`opt-level = 3` for dependencies even in dev, so `cargo test` is fast; do not
remove that block to "speed up compiles".

**The engine runs one generation at a time.** `Engine` holds one session under
a mutex; concurrent HTTP requests queue. Batching is a roadmap item, not a bug.

**A GGUF's tokenizer is rebuilt from its metadata.** `tokenizer.ggml.model`
is `llama` (SentencePiece, rebuilt as a Unigram model) or `gpt2` (byte-level
BPE, rebuilt from the merges). A GGUF with another tokenizer type loads its
weights fine and fails at the tokenizer with a message saying so.

**Runtimes started by infy are tracked by a pid file** under the data dir.
`infy runtime stop ollama` refuses to kill an ollama it did not start and tells
you how to stop it instead; a stray kill of the user's system service would be
worse than a refusal.

**CI runs the latest stable Rust, and clippy gains lints with every
release.** A push that is clean locally can fail `make lint` on the runner.
Before pushing from an older toolchain, run the runner's version:
`rustup toolchain install stable` then `cargo +stable clippy --workspace
--all-targets -- -D warnings`. Fix the lint rather than allowing it; every
one so far pointed at something worth changing.

**The Hugging Face hub is not reachable from every sandbox.** `infy pull`
fails with the proxy's status code and says so; nothing else in the system
needs the network.

## Session hygiene

| If this session… | Then update |
|---|---|
| changed layout, commands or a gotcha above | this file |
| changed a rule about how code is written | `GUIDELINES.md` |
| made a decision worth not re-litigating | `docs/DECISIONS.md` |
| shipped or changed a capability | `docs/product/CAPABILITIES.md`, same commit |
| changed what is next, or its order | `docs/product/ROADMAP.md`, with the reason |
| measured something | `docs/DECISIONS.md`, with the numbers |
| reached a wrong conclusion | correct it in `docs/DECISIONS.md` — do **not** delete the wrong one |

Record the measurement, not just the conclusion.

## This machine (the sandbox the scaffold was built in)

| | |
|---|---|
| Rust | 1.94.1 stable, cargo 1.94.1 |
| CPU / RAM | 4 cores, 15 GiB, no GPU |
| Network | crates.io and PyPI reachable; **huggingface.co and github LFS are not** (proxy 403) |
| Consequence | every model test uses a tiny random GGUF written by the test itself; no real checkpoint was run here -- see `docs/DECISIONS.md` |
