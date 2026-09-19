//! Pure logic: sampling, stop sequences, streaming deltas, the prompt-cache
//! plan, and chat-template rendering. Nothing here touches a model, a
//! tokenizer, a clock or a socket.

use std::cmp::Ordering;

use infy_kernel::{Error, Message, Result, SamplingParams, TokenId};

// --- randomness --------------------------------------------------------------

/// A small, fast, seedable generator (xorshift64*). Sampling needs a uniform
/// number per token and reproducibility from a seed; it does not need a
/// cryptographic source, and a dependency would be one more thing the crate
/// drags with it.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Zero is the one state xorshift cannot leave.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// --- sampling ----------------------------------------------------------------

/// Turn logits into a token.
///
/// In order: repetition penalty over `recent`, then temperature (zero means
/// greedy), then top-k, then top-p, then one draw using `u ∈ [0, 1)`. Pure:
/// the same logits, params and `u` always give the same token, which is what
/// makes a seeded run reproducible and a test a table.
pub fn sample(logits: &[f32], p: &SamplingParams, recent: &[TokenId], u: f32) -> TokenId {
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    let mut l: Vec<f32> = logits
        .iter()
        .map(|&x| if x.is_finite() { x } else { f32::NEG_INFINITY })
        .collect();

    if p.repetition_penalty != 1.0 {
        let window = recent.len().saturating_sub(p.repetition_window);
        for &t in &recent[window..] {
            if let Some(x) = l.get_mut(t as usize) {
                // The HF convention: divide a positive logit, multiply a
                // negative one; both push the token down.
                *x = if *x > 0.0 {
                    *x / p.repetition_penalty
                } else {
                    *x * p.repetition_penalty
                };
            }
        }
    }

    if p.temperature <= 0.0 {
        return argmax(&l);
    }

    let desc =
        |a: &(usize, f32), b: &(usize, f32)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
    let mut cand: Vec<(usize, f32)> = l
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x / p.temperature))
        .collect();
    if p.top_k > 0 && p.top_k < cand.len() {
        cand.select_nth_unstable_by(p.top_k, desc);
        cand.truncate(p.top_k);
    }
    let needs_order = p.top_p < 1.0;
    if needs_order {
        cand.sort_by(desc);
    }

    let max = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return argmax(&l);
    }
    let mut probs: Vec<f32> = cand.iter().map(|c| (c.1 - max).exp()).collect();
    let mut total: f32 = probs.iter().sum();

    if needs_order {
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (i, pr) in probs.iter().enumerate() {
            cum += pr / total;
            if cum >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        probs.truncate(keep);
        cand.truncate(keep);
        total = probs.iter().sum();
    }

    let target = u.clamp(0.0, 0.999_999) * total;
    let mut acc = 0.0;
    for (i, pr) in probs.iter().enumerate() {
        acc += pr;
        if acc > target {
            return cand[i].0 as TokenId;
        }
    }
    cand.last().map(|c| c.0 as TokenId).unwrap_or(0)
}

fn argmax(l: &[f32]) -> TokenId {
    let mut best = 0;
    for (i, &x) in l.iter().enumerate() {
        if x > l[best] {
            best = i;
        }
    }
    best as TokenId
}

// --- stop sequences ------------------------------------------------------------

/// What to do with the text not yet emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopCheck {
    /// Emit everything.
    NoMatch,
    /// Emit all but the last `n` bytes: they could be the start of a stop
    /// sequence that a later token completes.
    Partial(usize),
    /// A stop sequence begins at byte `i`. Emit `pending[..i]` and finish.
    Hit(usize),
}

/// Decide how much of `pending` is safe to emit given `stops`.
///
/// The held-back suffix is measured on character boundaries, so a
/// multi-byte character is never split, and the earliest hit wins when
/// several stop sequences appear.
pub fn check_stop(pending: &str, stops: &[String]) -> StopCheck {
    if stops.is_empty() || pending.is_empty() {
        return StopCheck::NoMatch;
    }
    let hit = stops.iter().filter_map(|s| pending.find(s.as_str())).min();
    if let Some(i) = hit {
        return StopCheck::Hit(i);
    }
    for (idx, _) in pending.char_indices() {
        let suffix = &pending[idx..];
        if stops
            .iter()
            .any(|s| s.len() > suffix.len() && s.starts_with(suffix))
        {
            return StopCheck::Partial(pending.len() - idx);
        }
    }
    StopCheck::NoMatch
}

// --- streaming -----------------------------------------------------------------

/// The text newly completed between two decodes of a growing token list.
///
/// `None` while the tail is an incomplete UTF-8 sequence (a byte-level
/// tokenizer can split a character across tokens, and the decoder renders
/// the fragment as U+FFFD); the caller waits for the next token. When the
/// previous text is a prefix of the current, the delta is the rest; if a
/// tokenizer's normalisation ever rewrites earlier text, the common prefix
/// is kept and the rest emitted, so nothing is dropped and nothing repeats.
pub fn text_delta<'a>(prev: &str, cur: &'a str) -> Option<&'a str> {
    if cur.ends_with('\u{FFFD}') {
        return None;
    }
    if let Some(rest) = cur.strip_prefix(prev) {
        return Some(rest);
    }
    let common = prev
        .char_indices()
        .zip(cur.char_indices())
        .take_while(|((_, a), (_, b))| a == b)
        .last()
        .map(|((i, a), _)| i + a.len_utf8())
        .unwrap_or(0);
    Some(&cur[common..])
}

// --- prompt cache ----------------------------------------------------------------

/// How to reuse a session whose cache holds `cached` when the new prompt is
/// `prompt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    /// Keep this many cached positions; drop the rest.
    pub keep: usize,
    /// Feed `prompt[keep..]`.
    pub feed_from: usize,
}

/// Keep the longest common prefix -- except that at least one token is
/// always fed, because the logits for the next position only exist after a
/// feed. A prompt identical to the cache re-feeds its last token.
pub fn plan_prefill(cached: &[TokenId], prompt: &[TokenId]) -> PrefillPlan {
    let mut lcp = cached
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count();
    if lcp == prompt.len() {
        lcp = prompt.len().saturating_sub(1);
    }
    PrefillPlan {
        keep: lcp,
        feed_from: lcp,
    }
}

/// Whether a BOS should be prepended: only if the checkpoint wants one and
/// the prompt does not already begin with its text (a chat template often
/// splices `bos_token` in itself, and a doubled BOS measurably hurts).
pub fn needs_bos(prompt: &str, bos_text: Option<&str>, add_bos: bool) -> bool {
    if !add_bos {
        return false;
    }
    match bos_text {
        Some(b) if !b.is_empty() => !prompt.starts_with(b),
        _ => true,
    }
}

// --- chat templates --------------------------------------------------------------

/// Render a checkpoint's Jinja chat template the way `transformers` does:
/// `trim_blocks` and `lstrip_blocks` on, Python string methods available
/// (`.strip()`, `.split()`, `.startswith()`), `raise_exception` defined,
/// `tools` absent, and `add_generation_prompt` true so the model's turn is
/// opened.
pub fn render_chat_template(
    template: &str,
    messages: &[Message],
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<String> {
    use minijinja::{context, Environment, ErrorKind, Value};

    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
        },
    );
    let tmpl = env
        .template_from_str(template)
        .map_err(|e| Error::invalid(format!("chat template does not parse: {e}")))?;
    tmpl.render(context! {
        messages => messages,
        bos_token => bos_token.unwrap_or(""),
        eos_token => eos_token.unwrap_or(""),
        add_generation_prompt => true,
        tools => Value::from(()),
    })
    .map_err(|e| Error::invalid(format!("chat template failed: {e}")))
}

/// The fallback when a checkpoint carries no template: ChatML, which most
/// instruct models of the last two years understand. `<|im_end|>` is also
/// a stop sequence when this is used, because a base model may not have it
/// as a single token.
pub fn chatml(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str("<|im_start|>");
        out.push_str(m.role.as_str());
        out.push('\n');
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

pub const CHATML_STOP: &str = "<|im_end|>";

#[cfg(test)]
#[path = "logic_test.rs"]
mod tests;
