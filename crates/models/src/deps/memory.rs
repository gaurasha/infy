//! An in-memory [`WeightSource`]: a tiny model with seeded random weights,
//! shipped with the domain so its tests, and the engine's, need no file, no
//! download and no network. Also the thing `write_gguf` turns into a real
//! file for the round-trip test.

use std::collections::HashMap;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};

use infy_kernel::{Error, Result, Vocabulary};

use crate::deps::WeightSource;
use crate::logic::{vocabulary_to_metadata, Architecture, Metadata, ModelConfig, RopeScaling};

/// SplitMix64 with a Box–Muller normal: deterministic weights from a seed,
/// with no dependency beyond `std`.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self, std: f64) -> f32 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * std) as f32
    }
}

struct Stored {
    data: Vec<f32>,
    shape: Vec<usize>,
    dtype: GgmlDType,
}

/// A random model held in memory.
pub struct RandomModel {
    meta: Metadata,
    tensors: HashMap<String, Stored>,
}

impl RandomModel {
    /// The configuration [`tiny`](Self::tiny) uses: two layers, four heads
    /// over two KV heads, a 64-token vocabulary. Small enough that a forward
    /// pass is microseconds and big enough that grouped-query attention and
    /// the cache are exercised.
    pub fn tiny_config(architecture: Architecture) -> ModelConfig {
        ModelConfig {
            architecture,
            name: "tiny-random".into(),
            hidden_size: 32,
            intermediate_size: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            vocab_size: 64,
            context_length: 256,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            rope_scaling: RopeScaling::None,
        }
    }

    /// A tiny random Llama, quantised to `Q8_0` where a real file would be
    /// quantised (projections, embedding) and `F32` where it would not
    /// (norms, biases).
    pub fn tiny(seed: u64) -> Self {
        Self::new(
            &Self::tiny_config(Architecture::Llama),
            seed,
            GgmlDType::Q8_0,
        )
    }

    /// Random weights for any configuration. `dtype` is the quantisation of
    /// the matrices; block sizes must divide the matrix dimensions (32 for
    /// `Q8_0`/`Q4_0`, 256 for the K-quants).
    pub fn new(config: &ModelConfig, seed: u64, dtype: GgmlDType) -> Self {
        let mut rng = Rng(seed);
        let mut tensors = HashMap::new();
        let with_bias = config.architecture == Architecture::Qwen2;
        for spec in config.tensor_spec() {
            let is_norm = spec.name.ends_with("norm.weight");
            let is_bias = spec.name.ends_with(".bias");
            let is_head = spec.name == "output.weight";
            let is_freqs = spec.name == "rope_freqs.weight";
            // Tied head and no rope scaling, like most small models. Biases
            // only where the architecture has them.
            if is_head || is_freqs || (is_bias && !with_bias) {
                continue;
            }
            let n: usize = spec.shape.iter().product();
            let (data, dt) = if is_norm {
                (
                    (0..n).map(|_| 1.0 + rng.normal(0.05)).collect::<Vec<f32>>(),
                    GgmlDType::F32,
                )
            } else if is_bias {
                ((0..n).map(|_| rng.normal(0.02)).collect(), GgmlDType::F32)
            } else {
                ((0..n).map(|_| rng.normal(0.1)).collect(), dtype)
            };
            tensors.insert(
                spec.name,
                Stored {
                    data,
                    shape: spec.shape,
                    dtype: dt,
                },
            );
        }
        Self {
            meta: config.to_metadata(),
            tensors,
        }
    }

    /// Attach a vocabulary, as a real file would carry one.
    pub fn with_vocabulary(mut self, v: &Vocabulary) -> Self {
        self.meta.extend(vocabulary_to_metadata(v));
        self
    }

    /// Add or override a metadata entry.
    pub fn with_metadata(mut self, k: &str, v: crate::logic::MetaValue) -> Self {
        self.meta.insert(k.to_string(), v);
        self
    }
}

impl WeightSource for RandomModel {
    fn metadata(&self) -> &Metadata {
        &self.meta
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tensors.keys().cloned().collect();
        names.sort();
        names
    }

    fn tensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        let s = self
            .tensors
            .get(name)
            .ok_or_else(|| Error::not_found(format!("tensor {name:?} in the in-memory model")))?;
        let t = Tensor::from_slice(&s.data, s.shape.as_slice(), &Device::Cpu)
            .map_err(|e| Error::wrap(format!("building {name}"), e))?;
        QTensor::quantize_onto(&t, s.dtype, device)
            .map_err(|e| Error::wrap(format!("quantising {name} to {:?}", s.dtype), e))
    }
}
