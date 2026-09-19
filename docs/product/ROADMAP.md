# infy — roadmap

What comes next, in order, and why that order.

Why infy exists: [BRIEF.md](BRIEF.md) · What exists today: [CAPABILITIES.md](CAPABILITIES.md)

**This is a sequence, not a schedule.** No dates. Each phase names what it
ships, how we know it is finished, and what would make it the wrong thing to
build next.

---

## The ordering principle

**Each phase must make infy useful on its own, not merely closer to useful.**

The failure mode for a project like this is a year of beautiful infrastructure
that runs nothing anyone uses, because the payoff was always one phase away. So
every phase below ends in something you would miss if it were removed.

The corollary: **measure before optimising.** Every optimisation phase starts
by recording the number it intends to move, on the real path, in
`docs/DECISIONS.md`.

---

## v1 — Foundation ✅ scaffolded

**Goal.** Run a language model locally two ways through one interface, with
the structure that keeps every later optimisation cheap.

Native GGUF execution with our own Llama-family architecture and KV cache;
tokenizer and chat template from the file itself; sampling, streaming, stop
sequences and a prompt cache, all pure and table-tested; ollama, llama.cpp and
LM Studio as data-driven integrations; an OpenAI-compatible server and a REPL;
usage and timing on every call; and the architecture rules that keep the rest
of this list cheap to build.

**What it proved.** That a domain can be tested with no weights, no network and
no GPU; that the model boundary at logits keeps the engine pure; that one wire
format serves both our server and our clients to other runtimes.

**What it did not prove.** Numerical agreement with a reference on a real
checkpoint — the build sandbox cannot reach the hub. **The first task on a real
machine is `infy pull` a small instruct model, run it natively and through
ollama, and compare.**

---

## v1.1 — Prove it on real weights

**Goal.** Turn "the invariants hold" into "the numbers match".

**Ships**

- A real small checkpoint (SmolLM2-135M-Instruct or Qwen2.5-0.5B-Instruct GGUF)
  run natively and via ollama, same prompt, same seed, greedy: outputs compared
- CUDA and Metal exercised, tokens per second recorded per device
- Any architecture bug that surfaces, fixed with the invariant test that would
  have caught it

**Done when** DECISIONS.md has a table of tok/s and time-to-first-token for at
least one model on at least one GPU, and the native and reference outputs agree.

---

## v2 — Throughput

**Goal.** Make the native path the fast path.

**Ships**

- Batching: several sessions in flight, scheduled per step
- Continuous prefill/decode scheduling with a fairness rule
- Prompt cache across requests (prefix trie), not only across turns of one chat
- safetensors loading with on-the-fly quantisation, so a model does not have to
  exist as a GGUF first

**Why here.** Everything above is a change to `engine` and `models` with an
invariant test beside it; nothing above needs a new domain.

**Done when** the server sustains N concurrent chats at a measured fraction of
single-stream throughput, and that fraction is in DECISIONS.md.

---

## v3 — Smarter decoding

**Goal.** Fewer forward passes per token.

**Ships**

- Speculative decoding with a small draft model
- Structured / constrained output (grammar-guided sampling)
- More architectures behind the `CausalLm` trait: Gemma, Phi, Mixtral

**Done when** speculative decoding shows a measured speedup on a real pair of
models, and constrained output produces valid JSON on demand.

---

## v4 — Voice

**Goal.** Speech in, speech out, through the same engine.

Speech-to-text and text-to-speech as new domains with the same shape; audio as
a stream type alongside tokens. Nothing here exists yet — this phase starts by
measuring whether a local STT model and the chat model can share the device.

---

## v5 — Image and video

**Goal.** Multimodal models natively.

Vision encoders as a domain; image tokens entering the same engine; diffusion
as a separate execution path with its own cache story. Deliberately vague at
this distance.

---

## v6 — Training and research

**Goal.** The same loaders, architectures and devices, run backwards.

Fine-tuning (LoRA first), then full training on small models; a research
harness for trying architectures on the pieces inference already proved. This
is last not because it matters least but because every piece under it must be
correct first — a training loop over a subtly wrong architecture is the most
expensive way to find the bug.

---

## Things deliberately not on this list

| | Why |
|---|---|
| A web UI | the OpenAI-compatible API means every existing chat UI already works against infy |
| A model zoo or curated catalogue | the hub and the runtimes' registries are the catalogue |
| Multi-user, hosting, accounts | single machine, single principal |
| Replacing ollama or llama.cpp | they are integrations; the native path exists to own optimisations, not to win a runtime war |

---

## How to change this document

When a phase ships, move it above the line with a ✅ and a short note on what it
actually proved — including where the plan was wrong. When a phase is reordered,
say what changed the ordering. The reasoning is the valuable part; a list of
features without it is a wish list.
