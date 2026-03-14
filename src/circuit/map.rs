use sha2::{Sha256, Digest};

/// Map arbitrary text to Lie algebra coefficients.
/// This is the chokepoint — every LLM output passes through here.
///
/// Two strategies:
///
/// 1. **Semantic (default)**: character n-gram embedding → deterministic projection → coefficients.
///    Semantically similar text maps to nearby algebra elements, preserving structure.
///
/// 2. **Hash fallback**: HMAC-style key derivation. Each coefficient gets its own
///    independent hash: H(text || domain || index). Fully deterministic, no semantic structure.
///
/// For production with real embeddings: text → sentence embedding → learned projection W → coefficients.
pub fn map_to_algebra(text: &str, algebra_dim: usize, scale: f64) -> Vec<f64> {
    map_to_algebra_semantic(text, algebra_dim, scale)
}

/// Hash-based mapping (original). Deterministic, semantically blind.
#[allow(dead_code)]
pub fn map_to_algebra_hash(text: &str, algebra_dim: usize, scale: f64) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(algebra_dim);
    for i in 0..algebra_dim {
        let mut hasher = Sha256::new();
        hasher.update(b"gravrail-map-to-algebra");
        hasher.update((i as u64).to_le_bytes());
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();

        let bytes: [u8; 4] = hash[..4].try_into().unwrap();
        let raw = u32::from_le_bytes(bytes) as f64 / u32::MAX as f64;
        let scaled = (raw * 2.0 - 1.0) * scale;
        coefficients.push(scaled);
    }
    coefficients
}

/// Semantic mapping: text → character n-gram embedding → projection → algebra coefficients.
///
/// Embeds text as a bag of character trigrams in a fixed-dimension space,
/// then projects to the algebra dimension using a deterministic random matrix.
/// Similar texts produce nearby algebra elements.
fn map_to_algebra_semantic(text: &str, algebra_dim: usize, scale: f64) -> Vec<f64> {
    // Step 1: Build a character n-gram embedding vector
    let embed_dim = 64; // internal embedding dimension
    let embedding = text_to_embedding(text, embed_dim);

    // Step 2: Project embedding to algebra dimension via deterministic matrix
    let coefficients = project_embedding(&embedding, algebra_dim);

    // Step 3: Normalize and scale
    let norm = coefficients.iter().map(|c| c * c).sum::<f64>().sqrt();
    if norm < 1e-12 {
        // Zero-ish embedding; fall back to hash
        return map_to_algebra_hash(text, algebra_dim, scale);
    }

    coefficients.iter()
        .map(|c| c / norm * scale)
        .collect()
}

/// Convert text to a fixed-dimension embedding using character n-grams.
///
/// Each character trigram hashes to a position in the embedding vector,
/// and we accumulate counts. This gives a bag-of-ngrams representation
/// that preserves lexical similarity.
fn text_to_embedding(text: &str, dim: usize) -> Vec<f64> {
    let mut embedding = vec![0.0f64; dim];
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();

    if chars.is_empty() {
        return embedding;
    }

    // Unigrams
    for &ch in &chars {
        let idx = (ch as usize) % dim;
        embedding[idx] += 1.0;
    }

    // Bigrams
    for window in chars.windows(2) {
        let hash = ((window[0] as u64).wrapping_mul(31) + window[1] as u64) as usize % dim;
        embedding[hash] += 2.0; // weight bigrams more
    }

    // Trigrams
    for window in chars.windows(3) {
        let hash = ((window[0] as u64).wrapping_mul(997)
            + (window[1] as u64).wrapping_mul(31)
            + window[2] as u64) as usize % dim;
        embedding[hash] += 3.0; // weight trigrams most
    }

    // Normalize to unit vector
    let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in embedding.iter_mut() {
            *x /= norm;
        }
    }

    embedding
}

/// Project a high-dimensional embedding to algebra dimension using
/// a deterministic pseudo-random projection matrix (Johnson-Lindenstrauss).
///
/// The matrix entries are derived from SHA256 for reproducibility.
fn project_embedding(embedding: &[f64], algebra_dim: usize) -> Vec<f64> {
    let embed_dim = embedding.len();
    let mut result = vec![0.0f64; algebra_dim];

    for i in 0..algebra_dim {
        for j in 0..embed_dim {
            // Deterministic matrix entry W[i][j]
            let mut hasher = Sha256::new();
            hasher.update(b"gravrail-projection-matrix");
            hasher.update((i as u64).to_le_bytes());
            hasher.update((j as u64).to_le_bytes());
            let hash = hasher.finalize();

            let bytes: [u8; 4] = hash[..4].try_into().unwrap();
            let w = (u32::from_le_bytes(bytes) as f64 / u32::MAX as f64) * 2.0 - 1.0; // [-1, 1]

            result[i] += w * embedding[j];
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_to_algebra_deterministic() {
        let a = map_to_algebra("hello world", 3, 1.0);
        let b = map_to_algebra("hello world", 3, 1.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_map_to_algebra_dimension() {
        let coeffs = map_to_algebra("test", 6, 1.0);
        assert_eq!(coeffs.len(), 6);
    }

    #[test]
    fn test_similar_texts_nearby() {
        let a = map_to_algebra("move left slowly", 3, 1.0);
        let b = map_to_algebra("move left quickly", 3, 1.0);
        let c = map_to_algebra("something completely different like fish", 3, 1.0);

        // Distance between similar texts should be smaller
        let dist_ab: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();
        let dist_ac: f64 = a.iter().zip(c.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt();

        assert!(dist_ab < dist_ac,
            "Similar texts should be closer: d(a,b)={} vs d(a,c)={}", dist_ab, dist_ac);
    }

    #[test]
    fn test_scale_factor() {
        let coeffs = map_to_algebra("test", 3, 2.0);
        let norm: f64 = coeffs.iter().map(|c| c * c).sum::<f64>().sqrt();
        assert!((norm - 2.0).abs() < 0.1, "Norm should be ~scale: got {}", norm);
    }

    #[test]
    fn test_empty_text_fallback() {
        let coeffs = map_to_algebra("", 3, 1.0);
        assert_eq!(coeffs.len(), 3);
        // Should still produce valid coefficients (hash fallback)
    }

    #[test]
    fn test_hash_fallback_matches_original() {
        let a = map_to_algebra_hash("test", 3, 1.0);
        assert_eq!(a.len(), 3);
        // Values should be in [-1, 1]
        for c in &a {
            assert!(*c >= -1.0 && *c <= 1.0);
        }
    }
}
