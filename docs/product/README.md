# docs/product/

What infy is supposed to be, what it can do today, and what comes next.

This folder is the **product** record. Engineering decisions — why Rust, why
candle, why GGUF first, why one session — live in
[../DECISIONS.md](../DECISIONS.md). If a question is "what should this do?", it
belongs here; if it is "why is it built that way?", it belongs there.

| File | Holds | Answers |
|---|---|---|
| [BRIEF.md](BRIEF.md) | the product definition | what is infy for, who for, what is it *not* |
| [CAPABILITIES.md](CAPABILITIES.md) | the feature inventory | can infy do X yet? |
| [ROADMAP.md](ROADMAP.md) | phases, in order, with reasoning | what is next, and why that next |

## Keeping it current

**`CAPABILITIES.md` changes in the same commit as the code.** It is the answer
to "can infy do X?", and a stale answer is worse than no answer — it makes
someone build something that already exists, or plan around something that does
not.

**`ROADMAP.md` records why the order changed**, not just the new order. A
roadmap without its reasoning is a wish list, and the reasoning is what stops
the same debate happening twice.

**`BRIEF.md` should change rarely.** If it is changing often, the product is
still being decided rather than built — which is worth noticing.
