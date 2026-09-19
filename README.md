# infy

A local inference engine. One Rust binary, your models, your machine.

infy is the layer where everything about running models locally lives:
loading them, caching, new architectures, and -- later -- training and research.
**v1 is text only**: language models, a chat REPL, and an OpenAI-compatible
HTTP server. Voice, image, video and training follow, and the structure is laid
out so they land as new domains rather than rewrites.

**What infy is for:** [docs/product/BRIEF.md](docs/product/BRIEF.md) ·
**what it can do today:** [docs/product/CAPABILITIES.md](docs/product/CAPABILITIES.md) ·
**what is next:** [docs/product/ROADMAP.md](docs/product/ROADMAP.md)

How to work here: [AGENT.md](AGENT.md) · How to write code: [GUIDELINES.md](GUIDELINES.md)
· Why it is shaped this way: [docs/DECISIONS.md](docs/DECISIONS.md)

## Two ways to run a model

**Natively.** A `.gguf` file is executed on Rust primitives: infy's own
Llama-family architecture and KV cache over candle's quantised kernels. This is
the path every optimisation in this repository lands on, and the one you own
end to end.

```bash
infy pull hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF        # picks the Q4_K_M
infy chat --model hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

**Through a runtime you already have.** ollama, llama.cpp and LM Studio are
detected on the machine (or pointed at with a flag), started and stopped, and
driven over their OpenAI-compatible APIs.

```bash
infy runtime status                                 # what is installed, what is running
infy runtime start ollama
infy chat --model ollama:llama3.2:3b
```

Both paths serve the same HTTP API:

```bash
infy serve --model hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF   # http://127.0.0.1:8321/v1
```

## Quick start

```bash
make check      # build, tests, lint, format, architecture rules
make chat ARGS="--model ./some-model.gguf"
```

Run `make` on its own to see every target.

## Layout

```
crates/
  kernel/    the types every layer names. serde only
  wire/      the OpenAI-compatible protocol. kernel + serde
  models/    architectures on Rust primitives: GGUF → Llama family, KV cache
  engine/    the generation loop: tokenise, prefill, decode, sample, stream, cache
  runtime/   ollama, llama.cpp, LM Studio -- as data, not code paths
  hub/       where models come from: refs, GGUF selection, download, catalogue
  server/    the HTTP API
  infy/      the composition root and CLI; the only crate importing everything
```

## Why it is built this way

**A domain crate can be dragged into another project and work.** Every domain
declares what it needs as traits, ships an in-memory implementation beside the
real one, and depends on no sibling. `engine` does not depend on `models` even
though running models is its whole job -- which is why its tests run with no
weights, no tokenizer file and no GPU. This is enforced by `make check-arch`
and `make check-portable`, not by good intentions.

**The model boundary is "tokens in, logits out".** The engine never sees a
tensor. Sampling, stop sequences, UTF-8-safe streaming and the prompt cache are
pure functions tested by tables, and they behave identically whether the logits
came from a GPU or from the test model.

**External runtimes are specs.** ollama, llama.cpp and LM Studio differ by a
binary name, a port, a start command and a model-list endpoint. Those are fields
of a table; the detect/start/stop/chat code is written once. Adding a runtime
is adding a row.

**Correctness is proven by invariants.** The KV cache is right if feeding
tokens one at a time gives the same logits as feeding them all at once. The
loader is right if a model written to disk reads back with identical logits.
Those are tests in the suite; "the text looked coherent" is not.

**Every generation records what it cost.** Tokens in, tokens out, tokens served
from cache, prefill time, time to first token, decode tokens per second -- on
every call, because those are the numbers every optimisation here is judged by.

**Failures say what is wrong and what to do.** ollama not installed, model file
missing, unsupported architecture, prompt longer than the context: each is a
specific message with the fix where one exists.

## Status

- [x] Native GGUF execution: Llama-family architecture, GQA, RoPE, growable KV cache
- [x] Tokenizer and chat template rebuilt from GGUF metadata
- [x] Sampling: temperature, top-k, top-p, repetition penalty, seeded and reproducible
- [x] Streaming with UTF-8-safe deltas and stop sequences held back correctly
- [x] Prompt cache: a multi-turn chat re-feeds only the new tokens
- [x] ollama, llama.cpp and LM Studio: detect, start, stop, install hint, list, chat
- [x] OpenAI-compatible server: chat completions (streaming and not), completions, models
- [x] Usage, timing and the exact prompt as sent on every call
- [ ] A real checkpoint run end to end (the build sandbox cannot reach the hub -- see DECISIONS.md)
- [ ] CUDA and Metal devices (wired through features, not yet exercised)
- [ ] Batching and concurrent sessions
- [ ] More architectures, safetensors loading, speculative decoding
- [ ] Voice, image, video; training
