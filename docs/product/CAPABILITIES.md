# infy — capabilities

The complete feature inventory: what exists, what is partial, what is only a
declared interface, and what is not built yet.

Why infy exists: [BRIEF.md](BRIEF.md) · When things land: [ROADMAP.md](ROADMAP.md)

**Keep this file current.** It is the answer to "can infy do X?", and a stale
answer is worse than no answer. When a capability changes state, change it here
in the same commit.

## Status legend

| | Meaning |
|---|---|
| ✅ **Shipped** | built, tested, and working end to end |
| 🟡 **Partial** | usable but incomplete; the gap is named |
| 🔵 **Interface only** | the type exists so the abstraction is right; no implementation |
| ⬜ **Planned** | committed to, with a roadmap phase |
| 💭 **Idea** | wanted, not yet designed |

---

## 1. Native execution (`models`, `engine`)

| Capability | Status | Where |
|---|---|---|
| Load a GGUF: metadata, quantised tensors, memory-mapped | ✅ | `models/deps/gguf.rs` |
| Llama-family architecture (Llama 2/3, Mistral, Qwen2, SmolLM, TinyLlama) | ✅ | `models/src/llama.rs` |
| Grouped-query attention, RoPE, RMSNorm, SwiGLU | ✅ | same |
| Llama 3 RoPE scaling | ✅ | `models/src/logic.rs` |
| KV cache: growable, truncatable, per layer | ✅ | `models/src/cache.rs` |
| Quantised matmul on CPU (Q4_0 … Q8_0, K-quants) | ✅ | candle `QMatMul` |
| CUDA / Metal devices | 🟡 | wired through cargo features; not exercised in the sandbox |
| Tokenizer rebuilt from GGUF metadata (`llama` and `gpt2` types) | ✅ | `engine/deps/tokenizers.rs` |
| Chat template from GGUF metadata (Jinja) with ChatML fallback | ✅ | `engine/src/logic.rs` |
| Sampling: greedy, temperature, top-k, top-p, repetition penalty | ✅ | `engine/src/logic.rs` |
| Reproducible generation from a seed | ✅ | same |
| Stop sequences, held back across token boundaries | ✅ | same |
| UTF-8-safe streaming of byte-level tokens | ✅ | same |
| Prompt cache: multi-turn chat feeds only the new tokens | ✅ | `engine/src/service.rs` |
| Cancellation between tokens, partial output kept | ✅ | `kernel::Cancel` |
| Usage and timing on every generation | ✅ | `kernel::Completion` |
| Numerical agreement with a reference implementation on a real checkpoint | ⬜ | needs hub access — see DECISIONS.md |
| Batching / concurrent sessions | ⬜ | phase 2 |
| safetensors loading | ⬜ | phase 2 |
| Speculative decoding | ⬜ | phase 3 |
| Other architectures (Gemma, Phi, Mixtral) | ⬜ | `CausalLm` trait reserved |

---

## 2. External runtimes (`runtime`)

| Capability | Status | Where |
|---|---|---|
| ollama: detect binary and server, start, stop, list models, chat | ✅ | `runtime/src/logic.rs` spec |
| llama.cpp `llama-server`: detect, start with a GGUF, stop, list, chat | ✅ | same |
| LM Studio `lms`: detect, start, stop, list, chat | ✅ | same |
| User-provided binary path and URL per runtime | ✅ | `--runtime-bin`, `--runtime-url` |
| Install: prints the official command, runs it with `--yes` | ✅ | `runtime/src/service.rs` |
| Streaming chat over OpenAI-compatible SSE with usage | ✅ | `runtime/deps/system.rs` |
| Pull a model through the runtime (`ollama pull`) | ✅ | `infy pull ollama:...` |
| Refuses to stop a runtime it did not start, names the right command | ✅ | pid file under the data dir |
| Health shown in `infy models` and `/health` | ✅ | |
| Native ollama `/api` features (keep-alive, options) | ⬜ | |

---

## 3. Models from somewhere (`hub`)

| Capability | Status | Where |
|---|---|---|
| Model refs: `hf:owner/repo`, `:QUANT`, `/file.gguf`, local name, path, `runtime:model` | ✅ | `hub/src/logic.rs` |
| Choose a GGUF from a repo listing (quant tag, Q4_K_M default, single file) | ✅ | same |
| Download into the models directory with resume via the hub client | ✅ | `hub/deps/hfhub.rs` |
| Local catalogue: what is in the models directory | ✅ | `hub/src/service.rs` |
| Sharded GGUFs (`-00001-of-00003`) | ⬜ | |
| ollama registry as a source for native execution | 💭 | its blobs are GGUF |

---

## 4. Interface (`server`, `infy`)

| Capability | Status | Where |
|---|---|---|
| `POST /v1/chat/completions`, streaming and not | ✅ | `server/src/service.rs` |
| `POST /v1/completions` | ✅ | same |
| `GET /v1/models`, `GET /health` | ✅ | same |
| Client disconnect cancels the generation | ✅ | `Cancel` dropped with the stream |
| `infy chat` REPL with per-turn usage and timing | ✅ | `infy/src/commands/chat.rs` |
| `infy generate`, `infy serve`, `infy pull`, `infy models`, `infy runtime` | ✅ | `infy/src/commands/` |
| `--show-prompt`: the exact prompt as sent | ✅ | |
| Web UI | ⬜ | |
| Embeddings endpoint | ⬜ | |

---

## 5. Platform

| Capability | Status | Where |
|---|---|---|
| Single binary, no daemon required | ✅ | `crates/infy` |
| Every location configurable, nothing derived from the binary path | ✅ | `--data-dir`, `--models-dir` |
| `HF_TOKEN` read once and removed from the environment | ✅ | `infy/src/bootstrap.rs` |
| Structured logging | ✅ | `tracing` |
| Architecture rules enforced in `make check` | ✅ | `scripts/` |
| Voice, image, video, multimodal | ⬜ | phases 4–5 |
| Training | ⬜ | phase 6 |

---

## Summary

| Area | Shipped | Partial | Interface | Planned | Idea |
|---|---|---|---|---|---|
| Native execution | 15 | 1 | 0 | 5 | 0 |
| External runtimes | 9 | 0 | 0 | 1 | 0 |
| Models from somewhere | 4 | 0 | 0 | 1 | 1 |
| Interface | 7 | 0 | 0 | 2 | 0 |
| Platform | 5 | 0 | 0 | 2 | 0 |

The shape of that table is the point: **the seams are all in place and the
native path is real, but it has not yet been run against a real checkpoint on
a machine that can download one.** That is the first thing to do with this
repository on a real machine.
