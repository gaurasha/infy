use candle_core::quantized::QTensor;
use candle_core::Device;

use infy_kernel::Result;

use crate::logic::Metadata;

/// Where a model's metadata and weights come from. Implementations live in
/// `deps/`: a GGUF file, and a random tiny model built in memory that ships
/// with the domain so its tests need no file.
///
/// The domain asks for tensors by their GGUF names (`blk.0.attn_q.weight`),
/// quantised, on a device. It never opens a file itself, which is what lets a
/// second source -- safetensors, a network blob, a tensor built by a training
/// loop -- arrive as an adapter rather than a change to the architecture.
pub trait WeightSource {
    /// The flat key → value metadata: architecture, dimensions, vocabulary.
    fn metadata(&self) -> &Metadata;

    /// Every tensor name the source can provide.
    fn tensor_names(&self) -> Vec<String>;

    fn has_tensor(&self, name: &str) -> bool {
        self.tensor_names().iter().any(|n| n == name)
    }

    /// One tensor, quantised as stored (or as `F32` for norms and biases),
    /// placed on `device`.
    fn tensor(&self, name: &str, device: &Device) -> Result<QTensor>;
}

pub mod gguf;
pub mod memory;
