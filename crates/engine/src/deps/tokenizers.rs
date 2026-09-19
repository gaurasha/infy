//! The real tokenizer adapter, over the `tokenizers` crate.
//!
//! A GGUF carries its vocabulary as data, not as a `tokenizer.json`, so the
//! tokenizer is rebuilt from that data the way llama.cpp does: a
//! `tokenizer.ggml.model` of `gpt2` is byte-level BPE (vocabulary plus
//! merges, split by the pre-tokeniser regex its `pre` names), and `llama` is
//! SentencePiece (vocabulary plus scores, rebuilt as a Unigram model with
//! byte fallback). Control tokens become special added tokens so a chat
//! template's `<|im_start|>` is one token, not eleven.

use tokenizers::decoders::byte_fallback::ByteFallback;
use tokenizers::decoders::fuse::Fuse;
use tokenizers::decoders::strip::Strip;
use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::bpe::BPE;
use tokenizers::models::unigram::Unigram;
use tokenizers::normalizers::{Prepend, Replace};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::{AddedToken, NormalizerWrapper, SplitDelimiterBehavior, TokenizerBuilder};

use infy_kernel::{Error, Result, TokenId, Vocabulary};

use crate::deps::Tokenizer;

/// GGUF token types.
const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

/// The GPT-2 pre-tokeniser regex, which most byte-level BPE vocabularies use.
const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
/// Llama 3's: case-insensitive contractions, digits in groups of up to three.
const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// Qwen2's: like Llama 3 but single digits.
const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The split regex a `tokenizer.ggml.pre` value calls for. Unknown values
/// get GPT-2's, which is what llama.cpp also falls back to (with a warning).
pub fn pre_tokenizer_pattern(pre: Option<&str>) -> &'static str {
    match pre {
        Some("llama-bpe") | Some("llama3") => LLAMA3_PATTERN,
        Some("qwen2") | Some("deepseek-llm") | Some("deepseek-coder") => QWEN2_PATTERN,
        _ => GPT2_PATTERN,
    }
}

/// A tokenizer rebuilt from a checkpoint's vocabulary.
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    vocab: Vocabulary,
    add_bos: bool,
}

impl HfTokenizer {
    pub fn from_vocabulary(vocab: Vocabulary) -> Result<Self> {
        let inner = match vocab.model.as_str() {
            "gpt2" => build_bpe(&vocab)?,
            "llama" => build_unigram(&vocab)?,
            other => {
                return Err(Error::invalid(format!(
                    "tokenizer type {other:?} is not supported yet; infy rebuilds gpt2 (byte-level BPE) and llama (SentencePiece) vocabularies"
                )))
            }
        };
        let add_bos = vocab.add_bos_token.unwrap_or(vocab.model == "llama");
        Ok(Self {
            inner,
            vocab,
            add_bos,
        })
    }

    /// Load a `tokenizer.json`, for checkpoints that ship one.
    pub fn from_file(path: &std::path::Path, vocab: Vocabulary) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| Error::internal(format!("reading {}: {e}", path.display())))?;
        let add_bos = vocab.add_bos_token.unwrap_or(false);
        Ok(Self {
            inner,
            vocab,
            add_bos,
        })
    }

    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }
}

fn special_tokens(vocab: &Vocabulary) -> Vec<AddedToken> {
    vocab
        .tokens
        .iter()
        .zip(vocab.token_types.iter().chain(std::iter::repeat(&1)))
        .filter(|(_, &t)| t == TYPE_CONTROL || t == TYPE_USER_DEFINED)
        .map(|(tok, &t)| AddedToken::from(tok.clone(), t == TYPE_CONTROL).normalized(false))
        .collect()
}

fn build_bpe(vocab: &Vocabulary) -> Result<tokenizers::Tokenizer> {
    let vocab_map: tokenizers::models::bpe::Vocab = vocab
        .tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();
    let merges = vocab
        .merges
        .iter()
        .map(|m| {
            m.split_once(' ')
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .ok_or_else(|| Error::invalid(format!("malformed merge {m:?}")))
        })
        .collect::<Result<Vec<_>>>()?;
    let model = BPE::builder()
        .vocab_and_merges(vocab_map, merges)
        .ignore_merges(matches!(
            vocab.pre.as_deref(),
            Some("llama-bpe") | Some("llama3")
        ))
        .build()
        .map_err(|e| Error::internal(format!("building BPE tokenizer: {e}")))?;
    let split = Split::new(
        tokenizers::pre_tokenizers::split::SplitPattern::Regex(
            pre_tokenizer_pattern(vocab.pre.as_deref()).to_string(),
        ),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| Error::internal(format!("building pre-tokenizer: {e}")))?;
    let byte_level = ByteLevel::new(false, false, false);
    let pre = tokenizers::pre_tokenizers::sequence::Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ]);
    let mut tk = TokenizerBuilder::new()
        .with_model(model)
        .with_normalizer(None::<NormalizerWrapper>)
        .with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(pre)))
        .with_post_processor(None::<tokenizers::PostProcessorWrapper>)
        .with_decoder(Some(DecoderWrapper::ByteLevel(byte_level)))
        .build()
        .map_err(|e| Error::internal(format!("assembling tokenizer: {e}")))?;
    tk.add_special_tokens(special_tokens(vocab))
        .map_err(|e| Error::internal(format!("registering special tokens: {e}")))?;
    Ok(tk.into())
}

fn build_unigram(vocab: &Vocabulary) -> Result<tokenizers::Tokenizer> {
    if vocab.scores.len() != vocab.tokens.len() {
        return Err(Error::invalid(format!(
            "SentencePiece vocabulary has {} tokens but {} scores",
            vocab.tokens.len(),
            vocab.scores.len()
        )));
    }
    let entries: Vec<(String, f64)> = vocab
        .tokens
        .iter()
        .zip(&vocab.scores)
        .map(|(t, &s)| (t.clone(), s as f64))
        .collect();
    let unk = vocab.unk_token_id.unwrap_or(0) as usize;
    let model = Unigram::from(entries, Some(unk), true)
        .map_err(|e| Error::internal(format!("building SentencePiece tokenizer: {e}")))?;
    let normalizer = tokenizers::normalizers::Sequence::new(vec![
        NormalizerWrapper::Prepend(Prepend::new("▁".into())),
        NormalizerWrapper::Replace(
            Replace::new(" ", "▁").map_err(|e| Error::internal(format!("normalizer: {e}")))?,
        ),
    ]);
    let decoder = tokenizers::decoders::sequence::Sequence::new(vec![
        DecoderWrapper::Replace(
            Replace::new("▁", " ").map_err(|e| Error::internal(format!("decoder: {e}")))?,
        ),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
        DecoderWrapper::Strip(Strip::new(' ', 1, 0)),
    ]);
    let mut tk = TokenizerBuilder::new()
        .with_model(model)
        .with_normalizer(Some(NormalizerWrapper::Sequence(normalizer)))
        .with_pre_tokenizer(None::<PreTokenizerWrapper>)
        .with_post_processor(None::<tokenizers::PostProcessorWrapper>)
        .with_decoder(Some(DecoderWrapper::Sequence(decoder)))
        .build()
        .map_err(|e| Error::internal(format!("assembling tokenizer: {e}")))?;
    tk.add_special_tokens(special_tokens(vocab))
        .map_err(|e| Error::internal(format!("registering special tokens: {e}")))?;
    Ok(tk.into())
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<TokenId>> {
        self.inner
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::internal(format!("tokenising: {e}")))
    }

    fn decode(&self, tokens: &[TokenId]) -> Result<String> {
        self.inner
            .decode(tokens, true)
            .map_err(|e| Error::internal(format!("detokenising: {e}")))
    }

    fn bos_token(&self) -> Option<TokenId> {
        self.vocab.bos_token_id
    }

    fn add_bos(&self) -> bool {
        self.add_bos
    }

    fn eos_tokens(&self) -> Vec<TokenId> {
        self.vocab.eos_token_ids.clone()
    }

    fn chat_template(&self) -> Option<String> {
        self.vocab.chat_template.clone()
    }

    fn special_token_text(&self, id: TokenId) -> Option<String> {
        self.vocab.tokens.get(id as usize).cloned()
    }
}

#[cfg(test)]
#[path = "tokenizers_test.rs"]
mod tests;
