# Decisions

Choices worth not re-litigating, and the measurements behind them. Each is a
deliberate tradeoff, not an accident.

Rules and conventions live in [../GUIDELINES.md](../GUIDELINES.md); this file
records *why the system is shaped the way it is*.

| Decision | Why |
|---|---|
| Rust, one binary | inference is a systems problem: memory layout, SIMD, threads, no GC pauses mid-token. And the local-model ecosystem's best pieces (llama.cpp's kernels, candle, tokenizers) are C++/Rust, not Python |
| candle as the tensor substrate, not our own kernels and not llama.cpp bindings | a tensor library is the one thing not worth rewriting for v1; bindings would hand the architecture, cache and scheduling to llama.cpp, which is exactly the layer infy exists to own. The `engine::Model` boundary is at logits, so candle can be replaced without the engine noticing |
| GGUF first, safetensors later | GGUF is what the local ecosystem actually ships: one file, quantised, with the tokenizer and chat template inside. safetensors needs a separate tokenizer file, a config file and a quantiser before it is useful locally |
| Our own Llama implementation over `QMatMul`, not `candle-transformers` | the architecture, the KV cache and the attention path are the things this project is for. Depending on someone else's copy of them would make every optimisation a fork |
| "Tokens in, logits out" is the model boundary | the engine never sees a tensor, so sampling, stop handling, streaming and the prompt cache are pure and identical for a GPU model and the in-memory test model. It also means a second backend (llama.cpp via FFI, a remote worker) is a `Model` impl, not a rewrite |
| External runtimes as `RuntimeSpec` rows | ollama, llama.cpp and LM Studio differ by binary name, port, start command and models endpoint. Everything else -- detect, start, stop, chat -- is the same code. Adding a runtime is adding a spec |
| The OpenAI chat-completions protocol is the one wire format | our server speaks it, and every runtime we drive speaks it, so `infy-wire` serves both directions. One set of types, one set of tests |
| Each crate declares its own dependencies; no workspace inheritance | a crate whose manifest says `serde.workspace = true` cannot be dragged into another workspace. `Cargo.lock` still pins one version tree-wide |
| KV cache grows by doubling, is never preallocated to the context length | preallocating 32k positions for a 0.5B model costs ~800 MB of zeros on a laptop that may never see a prompt over 2k. Growth is amortised O(1) per token; `truncate` is O(1) because it only moves the length |
| One generation at a time, one session per engine | a local machine runs one model at full speed or two at half; batching is a roadmap item with real scheduling questions, not a mutex to remove |
| `HF_TOKEN` read at startup then removed from the environment | infy spawns `ollama serve` and `llama-server` on purpose; a token left in the environment is inherited by both and readable in `/proc/<pid>/environ` |
| `infy runtime stop` refuses to kill a runtime it did not start | killing the user's system ollama because infy was asked to "stop ollama" is a worse outcome than a refusal that names the right command |

## Spike findings

### 1. The build sandbox cannot reach the Hugging Face hub

`curl https://huggingface.co/api/models/...` through the sandbox's egress proxy
returns `CONNECT tunnel failed, response 403`; so does `download.pytorch.org`.
crates.io and PyPI are reachable. **Consequence: no real checkpoint was run
during the scaffold.** Every model test writes its own tiny random GGUF and
reads it back. What that proves and does not prove is stated under
*Verified* below rather than glossed over.

### 2. `cargo tree -e normal` is the right graph for the architecture check

`-e normal` excludes dev- and build-dependencies, so a test helper crate is
not counted as coupling; `--prefix none` gives one `name version` per line
including transitive edges. That is exactly the question "can this folder be
lifted out" asks, and it needs no JSON parsing, so the check runs anywhere
cargo does.

### 3. Tensor code must be optimised even in dev builds

A candle matmul at `opt-level = 0` is 20-50x slower than at `3`. With
`[profile.dev.package."*"] opt-level = 3` the dependency crates are optimised
once and cached; our own crates stay at `opt-level = 0` and debuggable. The
model tests run in seconds rather than minutes.

## Verified

Filled in as things are actually run. Each entry names what was run, on what,
and what it does *not* show.

| What | Where | Shows | Does not show |
|---|---|---|---|
| KV cache consistency: incremental logits == full-prefill logits on a random 2-layer Llama | `crates/models/src/service_test.rs` | the cache, RoPE offsets and causal mask are consistent with each other | that the architecture matches Hugging Face's numerically (needs a real checkpoint) |
| GGUF round trip: write a tiny model, read it back, identical logits | same | tensor naming, metadata parsing and quantised loading agree with the in-memory model | compatibility with a llama.cpp-produced file (needs one) |
| Tokenizer rebuilt from GGUF metadata round-trips text | `crates/engine/src/deps/tokenizers_test.rs` | the BPE / Unigram reconstruction is self-consistent | byte-exact agreement with `tokenizer.json` for a real model |

## Verified environment facts

| | |
|---|---|
| Rust | 1.94.1 stable, cargo 1.94.1, x86_64-unknown-linux-gnu |
| CPU / RAM | 4 cores, 15 GiB, no GPU |
| Reachable | index.crates.io, pypi.org |
| Not reachable | huggingface.co, download.pytorch.org, GitHub LFS (proxy 403) |
