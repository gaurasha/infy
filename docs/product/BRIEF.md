# infy — product brief

What infy is for, who it serves, and what it deliberately is not.

Feature inventory: [CAPABILITIES.md](CAPABILITIES.md) ·
What comes next: [ROADMAP.md](ROADMAP.md) ·
Engineering decisions: [../DECISIONS.md](../DECISIONS.md)

---

## The one-sentence version

**infy is the layer for everything about running models on your own machine** —
loading them, running them fast, caching, new architectures, and eventually
training and research — behind one binary and one API.

## The problem

Running a model locally today means choosing a runtime first and living with
its choices: ollama's model store and its API, llama.cpp's flags, LM Studio's
GUI. Each is excellent at what it does and none of them is a place *you* can
work on the engine. Want to try a new caching strategy, a new sampling scheme,
a new architecture? You are forking C++.

And the interesting work is exactly there. Quantisation, KV-cache policy,
speculative decoding, prompt caching across turns, batching — these are where
local inference gets faster and cheaper, and they are hidden inside runtimes
that are not designed to be opened.

**The gap is not a missing runtime. It is a missing place to do the work** —
one that still lets you use the runtimes you already have.

## What infy is

A Rust workspace and a binary, `infy`, that together are the place where local
model work happens.

| | |
|---|---|
| **Runs models natively** | a `.gguf` executed on Rust primitives: infy's own architecture and KV cache over quantised kernels. This is the path every optimisation lands on |
| **Drives the runtimes you have** | ollama, llama.cpp, LM Studio: detected, started, stopped and talked to. Data, not code paths |
| **One interface** | a chat REPL and an OpenAI-compatible server, identical whichever path served the model |
| **Honest about cost** | tokens in, out and from cache, prefill time, time to first token, tokens per second, on every call |
| **Built to be opened** | every piece is a portable crate with a pure core and an in-memory implementation. Trying an idea means editing one file and running one test |

## Who it is for

The person who wants to *work on* local inference, not only use it: run a
model tonight, and next month change how it is run. Single user, single
machine. No tenancy, no accounts, one principal.

## Principles

**Own the layer that matters.** The tensor library is a dependency. The
architecture, the cache, the sampler, the scheduler and the loaders are ours.
That line is drawn on purpose and it is where the project earns its keep.

**Use what is already there.** A machine with ollama on it should not need a
second copy of every model. infy detects and drives existing runtimes rather
than competing with them.

**Correctness is an invariant, not an impression.** A cache is right if it
produces the same logits as no cache. A loader is right if a round trip is
lossless. Every optimisation ships with the test that proves it changed nothing
but speed.

**Everything is measured.** Every generation records what it cost. An
optimisation that is not measured on the real path does not exist.

**Degrade, never wedge.** No GPU, no ollama, no network, a model too big for
memory: each is a normal state with a specific message and, where there is
one, the command that fixes it.

**Everything is data, not configuration.** A model exists because its file is
in the models directory. A runtime is a spec. There is no config file.

## What infy is not

- **Not a chat product.** The REPL and the server are interfaces to the engine,
  not the point.
- **Not a model zoo.** It downloads what you ask for from where it lives; it
  does not curate.
- **Not a replacement for ollama or llama.cpp.** It drives them, and it runs
  models itself when you want to own the path. Both are first class.
- **Not multi-user, not hosted.** Nothing leaves the machine unless you point it
  at something that does.
- **Not a training framework yet.** Training is on the roadmap because the
  loaders, tensors and architectures it needs are the same ones inference
  needs. It arrives as a domain, when the pieces under it are proven.

## How we will know it is working

| Signal | What it tells us |
|---|---|
| A model runs natively and the numbers match a known-good runtime | the architecture and cache are right |
| An optimisation lands as one crate change with an invariant test | the structure is doing its job |
| Someone uses the ollama path and the native path in the same session without noticing | the interface is the engine's, not a runtime's |
| The measurements in DECISIONS.md are read before an optimisation is attempted | the discipline is paying for itself |

The failure mode to watch for is the opposite: a beautiful architecture that
runs nothing real, measured against nothing.
