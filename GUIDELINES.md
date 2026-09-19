# GUIDELINES.md

How to write code in infy. Navigation and architecture: [AGENT.md](AGENT.md).

Every rule here has its reason attached, because a rule without its reason gets
discarded by the next person who finds it inconvenient.

---

## The deciding factor

**A domain crate can be dragged into another project and work out of the box.**

Everything below follows from that one test. When two designs are both
defensible, the one that keeps a domain portable wins.

This is not an aesthetic preference. It is the only property that reliably keeps
a system this size from becoming one tangled object where nothing can be
replaced, tested in isolation, or reasoned about alone. An inference engine
accumulates optimisations faster than any other kind of software -- a new
cache, a new quantisation, a new attention kernel every month -- and each one
is a temptation to reach across a boundary "just this once".

---

## Module shape

Every domain is one crate under `crates/`, and every domain has the same shape:

```
crates/engine/
  Cargo.toml         its OWN dependencies, explicit versions (see below)
  src/
    lib.rs           what this is, what it needs, portability status; the
                     public surface -- re-exports of types and constructors
    deps.rs          the traits this domain REQUIRES from outside
    logic.rs         pure business logic: no I/O, no clock, no randomness
    logic_test.rs    table-driven, no mocks
    service.rs       orchestration -- holds deps, does I/O, calls logic
    service_test.rs  runs against deps/memory
    deps/
      mod.rs
      memory.rs      in-memory adapter, shipped with the domain
      tokenizers.rs  a real adapter
```

**`deps.rs` declares, `deps/` implements.** A domain never reaches for a
concrete dependency. It names a trait describing what it needs, and is handed
something implementing it. The domain does not know, and must not care, whether
the thing it was handed talks to a GPU, to a process on port 11434, or to a
`HashMap` in memory.

**`deps/memory.rs` ships with the domain, not with the tests.** This is the rule
that makes the deciding factor real. A domain crate dropped into another
project arrives with a runnable implementation and a test suite that passes
immediately -- no fixtures to port, no mocks to rebuild, no "you also need
these four other crates" README step. For `models` the memory adapter is a
tiny random-weight model built in memory; for `runtime` it is a fake process
table and a fake HTTP server. Both are real implementations of the trait.

**Tests sit beside the file they test**, as `logic_test.rs` next to `logic.rs`,
pulled in with `#[cfg(test)] #[path = "logic_test.rs"] mod tests;`. Rust
allows tests inline, but a 400-line file that is 60% test is a file nobody
reads. Same crate, same visibility, separate file.

**Each crate declares its own dependencies with explicit versions.** No
`serde.workspace = true`. Workspace inheritance is convenient and it breaks the
deciding factor: a crate whose manifest says "ask the workspace" does not work
in a workspace that has never heard of it. `Cargo.lock` still pins one version
per crate across the whole tree, so nothing drifts. `make check-portable`
copies each domain into an empty workspace and runs its tests there, which is
how this is verified rather than assumed.

---

## Functional core, imperative shell

**`logic.rs` is pure.** Values in, values out. No I/O, no clock, no randomness
that is not passed in, no mutation of arguments, no globals.

This is where the business logic lives, and purity is not decoration -- it is
what makes the logic testable with **zero mocks**. A test for pure logic is a
table of inputs and expected outputs. It cannot be flaky, it cannot need a
fixture, and it does not break when an unrelated dependency changes shape. That
payoff is the entire reason for the split.

What "pure" means for an inference engine, concretely: sampling takes the
logits and a uniform random number and returns a token. Stop-sequence detection
takes the pending text and the stop list and returns how many bytes are safe to
emit. Prompt-cache planning takes the cached tokens and the new prompt and
returns what to truncate and what to feed. None of them touch a model, a
tokenizer, a socket or a clock, so all of them are tested by tables.

**`service.rs` is the shell.** It is allowed to be imperative: take a `&Cancel`,
call deps, handle errors, hand values to `logic`, write the result back. It
should read like plumbing. If `service.rs` starts containing decisions rather
than sequencing, those decisions belong in `logic.rs`.

**No `static mut`, no `lazy_static`/`OnceLock` state, no singletons.** Each one
silently couples the domain to a particular process, and each one breaks the
drag-and-drop test in a way that only shows up later, in something unrelated,
at the worst time.

**Time and randomness are injected.** `Clock` is declared in `deps.rs` like any
other dependency. The sampler's random source is a seeded generator constructed
from a seed the caller chose; a domain that calls `Instant::now()` or
`rand::random()` directly cannot be tested deterministically, and
deterministic tests are the only kind worth having for a token sampler.

**Loggers and settings are passed in, never read from the environment inside a
domain.** The moment a domain calls `std::env::var`, it has stopped being
portable: it now depends on an environment its new host has no idea it needs.
`crates/infy` is the only crate that reads flags, the environment or a `.env`
file, and it hands plain values down.

---

## Dependencies: the rule that makes it real

**A domain crate may depend on: the standard library, third-party crates,
its own modules, `infy-kernel` and `infy-wire`. Nothing else.**

**No domain depends on a sibling domain -- ever.** Where two domains must talk,
the *consumer* declares a narrow trait in its own `deps.rs`, and `crates/infy`
supplies the other domain's concrete type.

Rust has no structural interface satisfaction and the orphan rule forbids
`impl engine::Model for models::Llama` anywhere except in one of those two
crates. So the composition root wraps the concrete type in a newtype and
implements the consumer's trait on that. Ten lines, and it is the *only* place
where two domains meet. That is not a workaround; it is the seam made visible.

**The rule binds the domain crate, not its `deps/` adapters.** An adapter under
`deps/` exists precisely to bridge the domain to something concrete, so
`engine/src/deps/tokenizers.rs` may pull in the `tokenizers` crate and
`runtime/src/deps/system.rs` may spawn processes. That is the ports-and-adapters
boundary, and it is what keeps the domain itself portable: drop the folder into
another project, keep `deps/memory.rs`, and delete the adapters that do not
apply. Heavy adapters sit behind a cargo feature for exactly that reason.

`crates/infy` is the composition root and the **only** crate permitted to
depend on everything. That is its entire job.

Two shared leaves are sanctioned, because without them the rule above is
impossible to keep:

| Leaf | Holds | May depend on |
|---|---|---|
| `infy-kernel` | the types every layer names -- `TokenId`, `Message`, `Usage`, `SamplingParams`, `Completion`, `Vocabulary`, `Cancel`, `Error` | `serde` only |
| `infy-wire` | the OpenAI-compatible HTTP protocol: request, response and chunk types, error codes, SSE framing | `infy-kernel`, `serde` |

Both are **types and pure functions only**. The moment something in `kernel`
opens a file, spawns a thread, or takes a tensor type, every domain has a
transitive dependency on whatever it touched. `serde` is the one permitted
third-party crate, because a type that cannot be serialised is not a type
every layer can name.

**This is enforced, not documented.** `make check-arch` walks `cargo tree` for
each domain and fails on a sibling edge, direct or transitive, and fails if a
leaf grows a dependency. `make check-portable` copies each domain into an
empty workspace and runs its tests. A rule nobody can verify is a rule that is
already being broken.

---

## Traits

**Two live implementations, or no trait.**

A trait written against a single implementation gets baked in by the call
sites written against it. When the second implementation finally arrives, it is
contorted to fit a shape that was never designed for it -- and by then dozens of
callers depend on that shape.

Extracting a trait later in Rust is nearly free: the compiler finds every call
site for you. So **reserve the concept in a doc comment; add the trait when the
second implementation exists.**

`engine::Model` earns its trait on day one because the memory model and the
`models` crate are both real. `models::Llama` is a plain struct, because there
is one architecture; the `CausalLm` trait is reserved in its doc comment for the
day a second architecture lands.

---

## Errors

- **`kernel::Error` is the one error type that crosses a domain boundary.** Its
  variants are sentinels -- `NotFound`, `Invalid`, `Unavailable`, `Cancelled`,
  `Internal` -- so any domain and the wire layer can match on them without
  importing a sibling.
- **Wrap with context** and enough of it to locate the failure without a
  debugger: `Error::wrap("reading tensor blk.3.attn_q.weight", e)`. The
  original error stays reachable through `source()`.
- **Never panic across a domain boundary.** Domain crates carry
  `#![deny(clippy::unwrap_used, clippy::expect_used)]`; tests are exempt. A
  domain that panics into its caller is not embeddable -- its new host cannot
  contain the blast radius. Tensor shape mismatches are errors, not asserts.
- **`&Cancel` is the first argument of every function that runs longer than a
  few milliseconds**, and it is honoured between tokens. This is a system built
  out of long-running generations; an uncancellable one is a hang waiting to
  happen.
- **`unsafe` lives only in `deps/` adapters**, with a `// SAFETY:` comment, and
  crates carry `#![deny(unsafe_code)]` with a single `#[allow]` at that site.
  Today the only one is memory-mapping a GGUF file.

---

## Tests

- **Every file carrying logic has its test beside it** -- `logic.rs` /
  `logic_test.rs`.
- **Pure logic is tested directly**, table-driven, with **no mocks**. If a test
  for `logic.rs` needs a mock, the function is not pure and belongs in
  `service.rs`.
- **Services are tested against `deps/memory`**, never a mocking framework. A
  hand-written fake you can read beats a generated mock you cannot.
- **A domain's tests must pass with only that crate plus the two leaves
  present.** That is the drag-and-drop test, and `make check-portable` runs it.
- **Model correctness is proven by invariants, not by eyeballing output.** The
  KV cache is correct if feeding tokens one at a time gives the same logits as
  feeding them all at once. A loader is correct if a model written to disk and
  read back gives the same logits as the one in memory. Those are tests; "the
  text looked coherent" is not.
- **Anything involving time is tested with the injected `Clock`**, so a test
  that measures latency or tokens per second is deterministic.

---

## Everything is data

**Only bootstrap settings come from flags or environment. Everything else is
data on disk or a row served over the API.**

| Layer | Source | Examples |
|---|---|---|
| Bootstrap | flags > env > `.env` > default | data dir, listen address, log level, device, `HF_TOKEN`, runtime binary and URL overrides |
| Everything else | files under the data dir, served over the API | the model catalogue, runtime state, and -- once there is a UI -- prompts, defaults and settings |

There is deliberately **no configuration file**. A config file immediately
becomes a second source of truth that has to be reconciled with what is on
disk, and reconciling two sources of truth is a permanent bug factory. A model
exists because its `.gguf` is in the models directory; a runtime is running
because its process answers on its port.

**External runtimes are specs, not code paths.** ollama, llama.cpp and LM Studio
differ by binary name, default port, start command and model-list endpoint.
Those are fields of a `RuntimeSpec`; the code that detects, starts, stops and
talks to a runtime is written once. Adding a runtime is adding a spec.

---

## Data and secrets

**No weights and no data ever enter the repo.** Models are hundreds of
megabytes to hundreds of gigabytes; the default models directory is outside the
checkout, and `.gitignore` blocks `*.gguf`, `*.safetensors`, `models/`, `data/`
and `.env` as a second line of defence.

**No path is derived from where the binary lives.** Every location is
configurable (`--data-dir`, `--models-dir`), so the binary can be installed
anywhere and pointed at data anywhere.

**`HF_TOKEN` is read at startup, then removed from the environment.** An
environment variable is inherited by every child process -- and this program
spawns child processes on purpose (`ollama serve`, `llama-server`) -- and is
readable in `/proc/<pid>/environ`. Removing it after load closes both exposures
at no cost. It is never logged and never appears in an error message.

---

## Failure discipline

**Every failure state gets a specific, visible message -- never a spinner that
never resolves.** ollama not installed, ollama installed but not running, a
model file missing, a GGUF whose architecture is not supported, a prompt longer
than the context window: each says exactly what is wrong and, where there is
one, the command that fixes it.

This is not polish. A system that fails opaquely is a system its owner stops
trusting, and then stops opening.

**Record tokens in, tokens out, tokens served from cache, and timing on every
generation.** Prefill time, time to first token, decode tokens per second.
These are the numbers every optimisation in this repository is judged by, so
they are measured on every call rather than in a benchmark someone remembers to
run.

**Show the exact prompt as sent.** After a chat template has been applied, with
special tokens, as a string. When an answer is bad, that is the only question
worth asking, and everything else is guesswork without it.

---

## Verify, don't assume

**Run `make check` before every commit.** It runs build, tests, clippy, format,
`check-arch` and `check-portable`.

**Check what a tool actually does rather than what it is documented to do.**
A quantised matmul that "supports bf16 input" may silently dequantise the whole
weight matrix on every call. Measure it.

**State what was verified and what was not.** "This failure mode was never
observed" is information; claiming it cannot happen is not. The end-to-end
runs in [docs/DECISIONS.md](docs/DECISIONS.md) say exactly which model, which
machine, and which numbers.

**Record a spike's finding, then delete the spike.** `_spike/` is gitignored
and temporary. The finding belongs in [docs/DECISIONS.md](docs/DECISIONS.md),
where it survives; the throwaway code does not.

---

## git

- Commit messages are imperative and say **why**, especially the non-obvious
  reason. If a bug is being fixed, describe the failure.
- **`.gitignore` has no inline comments.** A trailing `# ...` becomes part of the
  pattern and silently matches nothing. Comments go on their own line, and each
  non-obvious entry gets its reason.
- Verify a new ignore rule with `git check-ignore -q <path>` against a path that
  actually exists.
