//! The Llama-family decoder: RMSNorm → grouped-query attention with RoPE →
//! RMSNorm → SwiGLU feed-forward, repeated, then a final norm and the
//! language-model head. Weights stay quantised; activations are `dtype`.

use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::ops::{rms_norm, silu, softmax_last_dim};
use candle_nn::rotary_emb::{rope, rope_i};

use infy_kernel::{Error, Result, TokenId};

use crate::cache::KvCache;
use crate::deps::WeightSource;
use crate::logic::{causal_mask, rope_inv_freq, rope_tables, ModelConfig};

fn wrap<T>(r: candle_core::Result<T>, context: &str) -> Result<T> {
    r.map_err(|e| Error::wrap(context, e))
}

struct Layer {
    attn_norm: Tensor,
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    ffn_norm: Tensor,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

/// A loaded Llama-family model. Immutable once built; all per-sequence state
/// lives in a [`KvCache`], so one model serves many sessions.
///
/// Reserved: a `CausalLm` trait over `forward`/`new_cache` when a second
/// architecture exists.
pub struct Llama {
    config: ModelConfig,
    embed: candle_nn::Embedding,
    layers: Vec<Layer>,
    norm: Tensor,
    lm_head: QMatMul,
    inv_freq: Vec<f32>,
    freq_factors: Option<Vec<f32>>,
    device: Device,
    dtype: DType,
}

impl Llama {
    pub(crate) fn build(
        config: ModelConfig,
        source: &dyn WeightSource,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let present = source.tensor_names();
        let missing = config.missing_tensors(&present);
        if !missing.is_empty() {
            return Err(Error::invalid(format!(
                "checkpoint is missing {} tensor(s) the {} architecture needs, first: {}",
                missing.len(),
                config.architecture.gguf_name(),
                missing
                    .iter()
                    .take(3)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        let q = |name: &str| -> Result<QMatMul> {
            let t = source.tensor(name, device)?;
            wrap(QMatMul::from_qtensor(t), &format!("preparing {name}"))
        };
        let dense = |name: &str| -> Result<Tensor> {
            let t = source.tensor(name, device)?;
            let t = wrap(t.dequantize(device), &format!("dequantising {name}"))?;
            wrap(t.to_dtype(dtype), &format!("converting {name}"))
        };
        let dense_opt = |name: &str| -> Result<Option<Tensor>> {
            if source.has_tensor(name) {
                dense(name).map(Some)
            } else {
                Ok(None)
            }
        };

        let embed_table = dense("token_embd.weight")?;
        let embed = candle_nn::Embedding::new(embed_table, config.hidden_size);
        let lm_head = if source.has_tensor("output.weight") {
            q("output.weight")?
        } else {
            q("token_embd.weight")?
        };
        let norm = dense("output_norm.weight")?;
        let freq_factors = match dense_opt("rope_freqs.weight")? {
            Some(t) => Some(wrap(
                t.to_dtype(DType::F32)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>()),
                "reading rope_freqs.weight",
            )?),
            None => None,
        };

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let p = format!("blk.{i}");
            layers.push(Layer {
                attn_norm: dense(&format!("{p}.attn_norm.weight"))?,
                q: q(&format!("{p}.attn_q.weight"))?,
                k: q(&format!("{p}.attn_k.weight"))?,
                v: q(&format!("{p}.attn_v.weight"))?,
                o: q(&format!("{p}.attn_output.weight"))?,
                q_bias: dense_opt(&format!("{p}.attn_q.bias"))?,
                k_bias: dense_opt(&format!("{p}.attn_k.bias"))?,
                v_bias: dense_opt(&format!("{p}.attn_v.bias"))?,
                ffn_norm: dense(&format!("{p}.ffn_norm.weight"))?,
                gate: q(&format!("{p}.ffn_gate.weight"))?,
                up: q(&format!("{p}.ffn_up.weight"))?,
                down: q(&format!("{p}.ffn_down.weight"))?,
            });
        }

        Ok(Self {
            inv_freq: rope_inv_freq(config.head_dim, config.rope_theta),
            config,
            embed,
            layers,
            norm,
            lm_head,
            freq_factors,
            device: device.clone(),
            dtype,
        })
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// A fresh, empty cache sized for this model.
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(
            self.config.num_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            self.dtype,
            self.device.clone(),
        )
    }

    /// Feed `tokens` after whatever `cache` holds; return the logits for the
    /// last token, one `f32` per vocabulary entry. The cache advances.
    pub fn forward(&self, tokens: &[TokenId], cache: &mut KvCache) -> Result<Vec<f32>> {
        let logits = self.run(tokens, cache, false)?;
        wrap(
            logits
                .flatten_all()
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.to_vec1::<f32>()),
            "reading logits",
        )
    }

    /// Like [`forward`](Self::forward) but returns the logits at every fed
    /// position. What a cache-consistency test or a perplexity run needs.
    pub fn forward_all(&self, tokens: &[TokenId], cache: &mut KvCache) -> Result<Vec<Vec<f32>>> {
        let logits = self.run(tokens, cache, true)?;
        let t = tokens.len();
        wrap(
            logits
                .reshape((t, self.config.vocab_size))
                .and_then(|l| l.to_dtype(DType::F32))
                .and_then(|l| l.to_vec2::<f32>()),
            "reading logits",
        )
    }

    fn run(&self, tokens: &[TokenId], cache: &mut KvCache, all_positions: bool) -> Result<Tensor> {
        let t = tokens.len();
        if t == 0 {
            return Err(Error::invalid("forward called with no tokens"));
        }
        let offset = cache.len();
        if offset + t > self.config.context_length {
            return Err(Error::invalid(format!(
                "{} tokens ({offset} cached + {t} new) exceed the model's context length of {}",
                offset + t,
                self.config.context_length
            )));
        }
        for &id in tokens {
            if id as usize >= self.config.vocab_size {
                return Err(Error::invalid(format!(
                    "token id {id} is outside the vocabulary of {}",
                    self.config.vocab_size
                )));
            }
        }

        let ids = wrap(Tensor::new(tokens, &self.device), "building input ids")?;
        let mut x = wrap(
            self.embed.forward(&ids).and_then(|e| e.unsqueeze(0)),
            "embedding lookup",
        )?;

        let half = self.config.head_dim / 2;
        let (cos, sin) = rope_tables(
            &self.inv_freq,
            self.freq_factors.as_deref(),
            offset..offset + t,
            &self.config.rope_scaling,
        );
        let cos = wrap(
            Tensor::from_vec(cos, (t, half), &self.device).and_then(|c| c.to_dtype(self.dtype)),
            "building rope cos",
        )?;
        let sin = wrap(
            Tensor::from_vec(sin, (t, half), &self.device).and_then(|s| s.to_dtype(self.dtype)),
            "building rope sin",
        )?;
        let mask = if t > 1 {
            Some(wrap(
                Tensor::from_vec(causal_mask(t, offset), (t, offset + t), &self.device)
                    .and_then(|m| m.to_dtype(self.dtype)),
                "building causal mask",
            )?)
        } else {
            None
        };

        for (i, layer) in self.layers.iter().enumerate() {
            x = self.layer(layer, i, &x, &cos, &sin, mask.as_ref(), cache)?;
        }
        cache.advance(t);

        let x = wrap(
            rms_norm(&x, &self.norm, self.config.rms_norm_eps),
            "final norm",
        )?;
        let x = if all_positions {
            x
        } else {
            wrap(x.narrow(1, t - 1, 1), "selecting last position")?
        };
        wrap(self.lm_head.forward(&x), "lm head")
    }

    #[allow(clippy::too_many_arguments)]
    fn layer(
        &self,
        layer: &Layer,
        index: usize,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let cfg = &self.config;
        let ctx = |what: &str| format!("layer {index}: {what}");
        let (_, t, _) = wrap(x.dims3(), &ctx("input shape"))?;

        // --- attention ---
        let h = wrap(
            rms_norm(x, &layer.attn_norm, cfg.rms_norm_eps),
            &ctx("attention norm"),
        )?;
        let project =
            |m: &QMatMul, bias: &Option<Tensor>, heads: usize, what: &str| -> Result<Tensor> {
                let y = wrap(m.forward(&h), &ctx(what))?;
                let y = match bias {
                    Some(b) => wrap(y.broadcast_add(b), &ctx(what))?,
                    None => y,
                };
                wrap(
                    y.reshape((1, t, heads, cfg.head_dim))
                        .and_then(|y| y.transpose(1, 2))
                        .and_then(|y| y.contiguous()),
                    &ctx(what),
                )
            };
        let q = project(&layer.q, &layer.q_bias, cfg.num_heads, "q projection")?;
        let k = project(&layer.k, &layer.k_bias, cfg.num_kv_heads, "k projection")?;
        let v = project(&layer.v, &layer.v_bias, cfg.num_kv_heads, "v projection")?;

        let (q, k) = if cfg.rope_interleaved() {
            (
                wrap(rope_i(&q, cos, sin), &ctx("rope q"))?,
                wrap(rope_i(&k, cos, sin), &ctx("rope k"))?,
            )
        } else {
            (
                wrap(rope(&q, cos, sin), &ctx("rope q"))?,
                wrap(rope(&k, cos, sin), &ctx("rope k"))?,
            )
        };

        let (k, v) = cache.append(index, &k, &v)?;
        let rep = cfg.num_heads / cfg.num_kv_heads;
        let k = repeat_kv(&k, rep).map_err(|e| Error::wrap(ctx("expanding kv heads"), e))?;
        let v = repeat_kv(&v, rep).map_err(|e| Error::wrap(ctx("expanding kv heads"), e))?;

        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let att = wrap(
            k.t()
                .and_then(|kt| q.matmul(&kt))
                .and_then(|a| a.affine(scale, 0.0)),
            &ctx("attention scores"),
        )?;
        let att = match mask {
            Some(m) => wrap(att.broadcast_add(m), &ctx("causal mask"))?,
            None => att,
        };
        let att = wrap(softmax_last_dim(&att), &ctx("softmax"))?;
        let o = wrap(
            v.contiguous()
                .and_then(|v| att.matmul(&v))
                .and_then(|o| o.transpose(1, 2))
                .and_then(|o| o.reshape((1, t, cfg.num_heads * cfg.head_dim))),
            &ctx("attention output"),
        )?;
        let o = wrap(layer.o.forward(&o), &ctx("output projection"))?;
        let x = wrap(x + o, &ctx("residual"))?;

        // --- feed-forward ---
        let h = wrap(
            rms_norm(&x, &layer.ffn_norm, cfg.rms_norm_eps),
            &ctx("ffn norm"),
        )?;
        let g = wrap(
            layer.gate.forward(&h).and_then(|g| silu(&g)),
            &ctx("ffn gate"),
        )?;
        let u = wrap(layer.up.forward(&h), &ctx("ffn up"))?;
        let f = wrap(
            (g * u).and_then(|gu| layer.down.forward(&gu)),
            &ctx("ffn down"),
        )?;
        wrap(x + f, &ctx("residual"))
    }
}

/// Repeat each KV head `rep` times so grouped-query attention can use a plain
/// matmul: `(1, kv, l, d)` → `(1, kv × rep, l, d)`.
fn repeat_kv(x: &Tensor, rep: usize) -> candle_core::Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv, l, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, kv, rep, l, d))?
        .reshape((b, kv * rep, l, d))
}
