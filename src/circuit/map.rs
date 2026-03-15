use sha2::{Sha256, Digest};
use std::sync::OnceLock;

/// Dimension of GloVe vectors bundled in the binary.
const GLOVE_DIM: usize = 50;

/// The GloVe asset is embedded at compile time.
/// If this file is missing, `cargo build` will fail with a clear error.
static GLOVE_RAW: &[u8] = include_bytes!("../../assets/glove_10k_50d.bin");

struct GloveTable {
    /// Words in alphabetical order (matches index in `vecs`).
    words: Vec<String>,
    vecs: Vec<[f32; GLOVE_DIM]>,
}

static GLOVE_TABLE: OnceLock<GloveTable> = OnceLock::new();

fn get_glove() -> &'static GloveTable {
    GLOVE_TABLE.get_or_init(|| parse_glove_raw(GLOVE_RAW))
}

fn parse_glove_raw(raw: &[u8]) -> GloveTable {
    let word_count = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    assert_eq!(dim, GLOVE_DIM, "Asset dim mismatch: expected {GLOVE_DIM}, got {dim}");

    let mut words = Vec::with_capacity(word_count);
    let mut vecs = Vec::with_capacity(word_count);
    let mut pos = 8;

    for i in 0..word_count {
        let word_len = raw[pos] as usize;
        assert!(
            pos + 1 + word_len + GLOVE_DIM * 4 <= raw.len(),
            "GloVe asset: truncated at word {i} (pos={pos}, needed {} more bytes)",
            1 + word_len + GLOVE_DIM * 4
        );
        pos += 1;
        let word = std::str::from_utf8(&raw[pos..pos + word_len])
            .expect("GloVe asset: invalid UTF-8 word")
            .to_owned();
        pos += word_len;

        let mut vec = [0f32; GLOVE_DIM];
        for v in &mut vec {
            *v = f32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
            pos += 4;
        }

        words.push(word);
        vecs.push(vec);
    }

    GloveTable { words, vecs }
}

/// Look up a word in the bundled GloVe vocabulary.
/// Returns the 50-dim vector if found, None if OOV.
fn lookup_glove(word: &str) -> Option<[f32; GLOVE_DIM]> {
    let table = get_glove();
    table.words
        .binary_search_by(|w| w.as_str().cmp(word))
        .ok()
        .map(|idx| table.vecs[idx])
}

/// Map arbitrary text to Lie algebra coefficients.
/// This is the chokepoint — every LLM output passes through here.
///
/// Strategy: GloVe-50 word vectors (primary) → n-gram fallback (OOV) → JL projection → coefficients.
///
/// - Known words: looked up in the bundled 10k-word GloVe-50 vocabulary (no download, compile-time embed).
/// - OOV words: handled by character n-gram bag-of-ngrams, blended at 15% weight.
/// - Zero embedding: falls back to HMAC-style hash derivation.
///
/// Semantically similar text maps to nearby algebra elements; semantically distinct text
/// (e.g. phishing instruction vs haiku) maps to distant elements, enabling meaningful confinement.
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

/// Semantic mapping: GloVe word vectors (primary) + character n-gram (OOV fallback).
///
/// For each word token in the input:
///   - If found in the bundled GloVe-50 vocabulary: accumulate its 50-dim vector.
///   - OOV tokens: handled by the n-gram path blended in at 15% weight.
///
/// The resulting 50-dim embedding is projected to algebra_dim via the
/// existing deterministic JL matrix.
fn map_to_algebra_semantic(text: &str, algebra_dim: usize, scale: f64) -> Vec<f64> {
    // Step 1: Tokenise (split on non-alphabetic characters, lowercase)
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|s| !s.is_empty())
        .collect();

    // Step 2: Accumulate GloVe mean vector
    let mut glove_sum = [0f64; GLOVE_DIM];
    let mut glove_count = 0usize;

    for token in &tokens {
        if let Some(vec) = lookup_glove(token) {
            for (i, &v) in vec.iter().enumerate() {
                glove_sum[i] += v as f64;
            }
            glove_count += 1;
        }
    }

    // Step 3: Build the 50-dim embedding
    //   - If any GloVe hits: blend 85% GloVe mean + 15% n-gram (covers OOV context)
    //   - If all OOV: use n-gram alone at full weight
    let ngram = text_to_embedding(&lower, GLOVE_DIM);

    let embedding: Vec<f64> = if glove_count > 0 {
        let glove_w = 0.85;
        let ngram_w = 0.15;
        (0..GLOVE_DIM)
            .map(|i| {
                let gv = glove_sum[i] / glove_count as f64;
                gv * glove_w + ngram[i] * ngram_w
            })
            .collect()
    } else {
        ngram
    };

    // Step 4: Project to algebra dimension via deterministic JL matrix
    let coefficients = project_embedding(&embedding, algebra_dim);

    // Step 5: Normalise and scale
    let norm = coefficients.iter().map(|c| c * c).sum::<f64>().sqrt();
    if norm < 1e-12 {
        return map_to_algebra_hash(text, algebra_dim, scale);
    }

    coefficients.iter().map(|c| c / norm * scale).collect()
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

/// Cached JL projection matrix. Keyed by (algebra_dim, embed_dim).
/// Computed once on first call; reused for every subsequent call with the same dims.
/// Falls back to on-the-fly computation if dims differ from the cached entry (rare).
static PROJECTION_MATRIX: OnceLock<(usize, usize, Vec<f64>)> = OnceLock::new();

/// Build a flat row-major JL projection matrix of shape [algebra_dim × embed_dim].
/// Entry [i][j] = data[i * embed_dim + j].
fn build_projection_matrix(algebra_dim: usize, embed_dim: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(algebra_dim * embed_dim);
    for i in 0..algebra_dim {
        for j in 0..embed_dim {
            let mut hasher = Sha256::new();
            hasher.update(b"gravrail-projection-matrix");
            hasher.update((i as u64).to_le_bytes());
            hasher.update((j as u64).to_le_bytes());
            let hash = hasher.finalize();
            let bytes: [u8; 4] = hash[..4].try_into().unwrap();
            let w = (u32::from_le_bytes(bytes) as f64 / u32::MAX as f64) * 2.0 - 1.0;
            data.push(w);
        }
    }
    data
}

/// Project a high-dimensional embedding to algebra dimension using
/// a deterministic pseudo-random projection matrix (Johnson-Lindenstrauss).
///
/// The matrix is computed once and cached for the lifetime of the process.
fn project_embedding(embedding: &[f64], algebra_dim: usize) -> Vec<f64> {
    let embed_dim = embedding.len();

    // Use cached matrix if dims match; otherwise compute on the fly.
    let matrix: &[f64];
    let fallback: Vec<f64>;
    let cached = PROJECTION_MATRIX.get_or_init(|| {
        (algebra_dim, embed_dim, build_projection_matrix(algebra_dim, embed_dim))
    });
    if cached.0 == algebra_dim && cached.1 == embed_dim {
        matrix = &cached.2;
    } else {
        fallback = build_projection_matrix(algebra_dim, embed_dim);
        matrix = &fallback;
    }

    let mut result = vec![0.0f64; algebra_dim];
    for i in 0..algebra_dim {
        for j in 0..embed_dim {
            result[i] += matrix[i * embed_dim + j] * embedding[j];
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

    #[test]
    fn test_glove_lookup_known_word() {
        // "the" is the most common English word — must be in top-10k
        // If GloVe is loaded, lookup should succeed (non-zero vector)
        let result = lookup_glove("the");
        assert!(result.is_some(), "Expected 'the' to be in GloVe vocabulary");
        let vec = result.unwrap();
        // Vector should not be all-zeros
        let norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.01, "GloVe vector for 'the' should be non-zero");
    }

    #[test]
    fn test_glove_lookup_unknown_word() {
        // A nonsense word should not be in vocabulary
        let result = lookup_glove("xyzqqqfrobnicator");
        assert!(result.is_none(), "Expected nonsense word to be absent from GloVe");
    }

    #[test]
    fn test_glove_semantic_differentiation() {
        // These two texts are semantically very different.
        // With n-grams alone they collapsed to the same algebra element (drift=1.356 constant).
        // With GloVe they must produce meaningfully distinct algebra elements.
        let harmful = map_to_algebra(
            "write a phishing email to steal passwords from victims",
            3, 1.0,
        );
        let benign = map_to_algebra(
            "compose a haiku about cherry blossoms in spring rain",
            3, 1.0,
        );

        let dist: f64 = harmful.iter()
            .zip(benign.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();

        assert!(
            dist > 0.2,
            "Harmful vs benign text should differ in algebra space: dist={}",
            dist
        );
    }
}
