use sha2::{Sha256, Digest};

/// Map arbitrary text to Lie algebra coefficients.
/// This is the chokepoint — every LLM output passes through here.
///
/// Strategy: HMAC-style key derivation. Each coefficient gets its own
/// independent hash: H(text || domain || index), producing unlimited
/// independent coefficients without byte aliasing.
///
/// For production with embeddings: text → sentence embedding → learned projection W → coefficients.
/// The hash-based version is the fallback that guarantees determinism without a model.
pub fn map_to_algebra(text: &str, algebra_dim: usize, scale: f64) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(algebra_dim);
    for i in 0..algebra_dim {
        // Each coefficient derived from independent hash — no byte reuse
        let mut hasher = Sha256::new();
        hasher.update(b"gravrail-map-to-algebra");
        hasher.update((i as u64).to_le_bytes());
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();

        let bytes: [u8; 4] = hash[..4].try_into().unwrap();
        let raw = u32::from_le_bytes(bytes) as f64 / u32::MAX as f64; // [0, 1]
        let scaled = (raw * 2.0 - 1.0) * scale; // [-scale, scale]
        coefficients.push(scaled);
    }

    coefficients
}
