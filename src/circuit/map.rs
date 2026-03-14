use sha2::{Sha256, Digest};

/// Map arbitrary text to Lie algebra coefficients.
/// This is the chokepoint — every LLM output passes through here.
///
/// Strategy: hash-based deterministic projection.
/// Text → SHA256 → extract n f64 coefficients in [-scale, scale].
///
/// For production with embeddings: text → sentence embedding → learned projection W → coefficients.
/// The hash-based version is the fallback that guarantees determinism without a model.
pub fn map_to_algebra(text: &str, algebra_dim: usize, scale: f64) -> Vec<f64> {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();

    // Extract coefficients from hash bytes
    let mut coefficients = Vec::with_capacity(algebra_dim);
    for i in 0..algebra_dim {
        // Use 4 bytes per coefficient, cycling through hash
        let offset = (i * 4) % 28; // SHA256 = 32 bytes, use up to 28 for 7 f64s
        let bytes: [u8; 4] = [
            hash[offset % 32],
            hash[(offset + 1) % 32],
            hash[(offset + 2) % 32],
            hash[(offset + 3) % 32],
        ];
        let raw = u32::from_le_bytes(bytes) as f64 / u32::MAX as f64; // [0, 1]
        let scaled = (raw * 2.0 - 1.0) * scale; // [-scale, scale]
        coefficients.push(scaled);
    }

    coefficients
}
