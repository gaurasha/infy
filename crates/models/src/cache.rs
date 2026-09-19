//! The key/value cache: every position's K and V per layer, so a decode step
//! attends over the past without recomputing it.
//!
//! Storage grows by doubling and is never preallocated to the context
//! length: 32k positions of zeros for a model that will see a 2k prompt is
//! memory nobody asked for. `truncate` only moves the length, which is what
//! makes prompt-cache reuse across turns O(1).

use candle_core::{DType, Device, Tensor};

use infy_kernel::{Error, Result};

/// First capacity, positions. Doubles from here.
const INITIAL_CAPACITY: usize = 64;

struct LayerCache {
    /// `(1, num_kv_heads, capacity, head_dim)`; the first `len` positions are live.
    k: Tensor,
    v: Tensor,
    capacity: usize,
}

/// One sequence's cached keys and values across all layers.
pub struct KvCache {
    layers: Vec<Option<LayerCache>>,
    len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl KvCache {
    pub(crate) fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            len: 0,
            num_kv_heads,
            head_dim,
            dtype,
            device,
        }
    }

    /// Positions currently cached.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions allocated (the same for every layer once any has been used).
    pub fn capacity(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| l.capacity)
            .max()
            .unwrap_or(0)
    }

    /// Bytes held by the cache tensors, live and spare.
    pub fn memory_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|l| {
                2 * l.capacity * self.num_kv_heads * self.head_dim * self.dtype.size_in_bytes()
            })
            .sum()
    }

    /// Forget every position from `len` on. Storage is kept.
    pub fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }

    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Store this layer's new keys and values (`(1, num_kv_heads, t, head_dim)`)
    /// at the current end, and return the full live K and V
    /// (`(1, num_kv_heads, len + t, head_dim)`) for attention. `len` itself
    /// advances once per forward pass via [`advance`](Self::advance), after
    /// every layer has appended.
    pub(crate) fn append(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let t = k
            .dim(2)
            .map_err(|e| Error::wrap("kv cache: reading new key length", e))?;
        let needed = self.len + t;
        let (num_kv_heads, head_dim, dtype, device, len) = (
            self.num_kv_heads,
            self.head_dim,
            self.dtype,
            self.device.clone(),
            self.len,
        );
        let slot = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| Error::internal(format!("kv cache: layer {layer} out of range")))?;

        let entry = match slot {
            Some(entry) if entry.capacity >= needed => entry,
            existing => {
                let old_cap = existing.as_ref().map(|e| e.capacity).unwrap_or(0);
                let mut cap = old_cap.max(INITIAL_CAPACITY);
                while cap < needed {
                    cap *= 2;
                }
                let fresh = |old: Option<&Tensor>| -> Result<Tensor> {
                    let z = Tensor::zeros((1, num_kv_heads, cap, head_dim), dtype, &device)
                        .map_err(|e| Error::wrap("kv cache: allocating", e))?;
                    if let (Some(old), true) = (old, len > 0) {
                        let live = old
                            .narrow(2, 0, len)
                            .and_then(|x| x.contiguous())
                            .map_err(|e| Error::wrap("kv cache: reading live positions", e))?;
                        z.slice_set(&live, 2, 0)
                            .map_err(|e| Error::wrap("kv cache: copying into grown storage", e))?;
                    }
                    Ok(z)
                };
                let k_new = fresh(existing.as_ref().map(|e| &e.k))?;
                let v_new = fresh(existing.as_ref().map(|e| &e.v))?;
                *existing = Some(LayerCache {
                    k: k_new,
                    v: v_new,
                    capacity: cap,
                });
                match existing {
                    Some(e) => e,
                    None => return Err(Error::internal("kv cache: allocation vanished")),
                }
            }
        };

        entry
            .k
            .slice_set(k, 2, len)
            .map_err(|e| Error::wrap(format!("kv cache: writing keys for layer {layer}"), e))?;
        entry
            .v
            .slice_set(v, 2, len)
            .map_err(|e| Error::wrap(format!("kv cache: writing values for layer {layer}"), e))?;
        let k_all = entry
            .k
            .narrow(2, 0, needed)
            .map_err(|e| Error::wrap("kv cache: reading keys", e))?;
        let v_all = entry
            .v
            .narrow(2, 0, needed)
            .map_err(|e| Error::wrap("kv cache: reading values", e))?;
        Ok((k_all, v_all))
    }

    /// Commit `t` appended positions. Called once per forward pass.
    pub(crate) fn advance(&mut self, t: usize) {
        self.len += t;
    }
}
