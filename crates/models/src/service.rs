//! The imperative shell: pick a device, read the configuration, build the
//! model from a [`WeightSource`].

use candle_core::{DType, Device};

use infy_kernel::{Error, Result};

use crate::deps::WeightSource;
use crate::llama::Llama;
use crate::logic::ModelConfig;

/// Where and in what precision to run.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub device: Device,
    /// Activation dtype. Weights stay quantised regardless. `F32` is the safe
    /// default everywhere; `F16`/`BF16` are worth it only on an accelerator.
    pub dtype: DType,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            dtype: DType::F32,
        }
    }
}

/// Load a model. Reads the configuration, checks the architecture is one
/// this crate runs, checks every required tensor is present, and builds it.
pub fn load(source: &dyn WeightSource, opts: &LoadOptions) -> Result<Llama> {
    let config = ModelConfig::from_metadata(source.metadata())?;
    Llama::build(config, source, &opts.device, opts.dtype)
}

/// Resolve a device spec: `auto`, `cpu`, `cuda`, `cuda:N`, `metal`.
///
/// `auto` prefers CUDA, then Metal, then CPU. Asking for an accelerator this
/// binary was not built with is a specific error naming the feature flag,
/// not a silent fallback to CPU that leaves someone wondering why it is slow.
pub fn device(spec: &str) -> Result<Device> {
    use candle_core::utils::{cuda_is_available, metal_is_available};
    let wrap = |r: candle_core::Result<Device>, what: &str| {
        r.map_err(|e| Error::wrap(format!("opening {what} device"), e))
    };
    match spec {
        "cpu" => Ok(Device::Cpu),
        "auto" => {
            if cuda_is_available() {
                wrap(Device::new_cuda(0), "cuda:0")
            } else if metal_is_available() {
                wrap(Device::new_metal(0), "metal")
            } else {
                Ok(Device::Cpu)
            }
        }
        s if s == "cuda" || s.starts_with("cuda:") => {
            let ordinal = s
                .strip_prefix("cuda:")
                .map(|n| n.parse::<usize>())
                .transpose()
                .map_err(|_| Error::invalid(format!("device {s:?}: expected cuda or cuda:<n>")))?
                .unwrap_or(0);
            if !cuda_is_available() {
                return Err(Error::unavailable(
                    "this build has no CUDA support; rebuild with `--features cuda` on a machine with the CUDA toolkit",
                ));
            }
            wrap(Device::new_cuda(ordinal), s)
        }
        "metal" => {
            if !metal_is_available() {
                return Err(Error::unavailable(
                    "this build has no Metal support; rebuild with `--features metal` on macOS",
                ));
            }
            wrap(Device::new_metal(0), "metal")
        }
        other => Err(Error::invalid(format!(
            "unknown device {other:?}; use auto, cpu, cuda, cuda:<n> or metal"
        ))),
    }
}

#[cfg(test)]
#[path = "service_test.rs"]
mod tests;
