# GloVe-50 Bundled Embeddings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace character n-gram mapping in `src/circuit/map.rs` with bundled GloVe-50 word vectors so semantically distinct LLM outputs produce genuinely different SO(3)/SE(3) trajectories — no runtime download, fully offline.

**Architecture:** A Python script downloads GloVe 6B 50d once, filters to the top 10k words (sorted alphabetically), writes a compact binary asset to `assets/glove_10k_50d.bin`, and commits it to the repo. The Rust code embeds this asset via `include_bytes!` at compile time, parses it lazily into an `OnceLock`-backed lookup table on first call, and blends GloVe mean vectors with the existing n-gram fallback for OOV words.

**Tech Stack:** Python 3 (asset generation script), Rust std (`OnceLock`, `include_bytes!`), existing `sha2` crate (n-gram/hash fallback unchanged)

---

### Task 1: Generate the GloVe binary asset

**Files:**
- Create: `scripts/build_glove_asset.py`
- Create: `assets/glove_10k_50d.bin` (generated — must be committed to git)

**Step 1: Create the script**

```python
#!/usr/bin/env python3
"""Download GloVe 6B 50d, take top-10k words, write compact binary asset.

Binary format:
  [u32 LE: word_count][u32 LE: dim=50]
  for each word (alphabetical order):
    [u8: word_len][bytes: word (utf-8)][f32 LE × 50: vector]
"""
import struct
import urllib.request
import zipfile
import io
import os

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
TOP_N = 10_000
DIM = 50
OUT_FILE = "assets/glove_10k_50d.bin"


def main():
    os.makedirs("assets", exist_ok=True)

    print("Downloading GloVe 6B (822 MB)...")
    with urllib.request.urlopen(GLOVE_URL) as response:
        data = response.read()

    print("Extracting glove.6B.50d.txt...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        with zf.open("glove.6B.50d.txt") as f:
            lines = f.read().decode("utf-8").splitlines()

    # Take top N (file is frequency-sorted; first line = most common word)
    entries = []
    for line in lines[:TOP_N]:
        parts = line.split()
        word = parts[0]
        vec = [float(x) for x in parts[1:]]
        if len(vec) == DIM:
            entries.append((word, vec))

    # Sort alphabetically so Rust can binary-search
    entries.sort(key=lambda e: e[0])

    print(f"Writing {len(entries)} words to {OUT_FILE}...")
    with open(OUT_FILE, "wb") as f:
        f.write(struct.pack("<II", len(entries), DIM))
        for word, vec in entries:
            wb = word.encode("utf-8")
            f.write(struct.pack(f"<B{len(wb)}s", len(wb), wb))
            f.write(struct.pack(f"<{DIM}f", *vec))

    size_mb = os.path.getsize(OUT_FILE) / 1024 / 1024
    print(f"Done. {OUT_FILE}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
```

Save to `scripts/build_glove_asset.py`.

**Step 2: Run the script from the project root**

```bash
python3 scripts/build_glove_asset.py
```

Expected output:
```
Downloading GloVe 6B (822 MB)...
Extracting glove.6B.50d.txt...
Writing 10000 words to assets/glove_10k_50d.bin...
Done. assets/glove_10k_50d.bin: 2.0 MB
```

(Download takes a few minutes depending on connection.)

**Step 3: Verify the file exists and is ~2 MB**

```bash
ls -lh assets/glove_10k_50d.bin
```

Expected: file size between 1.8 MB and 2.2 MB.

**Step 4: Ensure assets/ is not gitignored**

```bash
git check-ignore -v assets/glove_10k_50d.bin
```

If it prints nothing, the file is not ignored — proceed. If it is ignored, add to `.gitignore`:
```
!assets/glove_10k_50d.bin
```

**Step 5: Commit the script and asset**

```bash
git add scripts/build_glove_asset.py assets/glove_10k_50d.bin
git commit -m "feat: add GloVe-50 top-10k asset and generation script"
```

---

### Task 2: Add the GloVe loader to map.rs (TDD)

**Files:**
- Modify: `src/circuit/map.rs`

#### Step 1: Write the failing tests first

Add these two tests to the `#[cfg(test)]` block at the bottom of `src/circuit/map.rs` (before the closing `}`):

```rust
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
```

**Step 2: Run to verify they fail**

```bash
cargo test test_glove_lookup --lib 2>&1 | head -20
```

Expected: FAIL with `cannot find function 'lookup_glove' in this scope` (or similar compile error).

**Step 3: Implement the GloVe loader**

Add the following to `src/circuit/map.rs`, right after the existing `use sha2::{Sha256, Digest};` line at the top:

```rust
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

    for _ in 0..word_count {
        let word_len = raw[pos] as usize;
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
```

**Step 4: Run the failing tests again — they should now pass**

```bash
cargo test test_glove_lookup --lib
```

Expected:
```
test tests::test_glove_lookup_known_word ... ok
test tests::test_glove_lookup_unknown_word ... ok
```

**Step 5: Run all existing tests to make sure nothing broke**

```bash
cargo test --lib
```

Expected: all existing tests pass.

**Step 6: Commit**

```bash
git add src/circuit/map.rs
git commit -m "feat: add GloVe-50 loader with OnceLock and binary search lookup"
```

---

### Task 3: Upgrade map_to_algebra_semantic to use GloVe vectors (TDD)

**Files:**
- Modify: `src/circuit/map.rs`

**Step 1: Write the failing semantic differentiation test**

Add to the `#[cfg(test)]` block in `src/circuit/map.rs`:

```rust
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
```

**Step 2: Run to verify it fails**

```bash
cargo test test_glove_semantic_differentiation --lib 2>&1 | tail -10
```

Expected: FAIL (the n-gram mapping produces dist ≈ 0 for these two inputs).

**Step 3: Replace map_to_algebra_semantic with the GloVe-aware version**

Replace the existing `fn map_to_algebra_semantic` function entirely (lines 43–61 in the original file) with:

```rust
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
```

Also update the `text_to_embedding` call signature — the function already takes `dim` as a param, so no change is needed there.

**Step 4: Run the new test**

```bash
cargo test test_glove_semantic_differentiation --lib
```

Expected: PASS.

**Step 5: Run all lib tests**

```bash
cargo test --lib
```

Expected: all tests pass including the pre-existing `test_similar_texts_nearby`, `test_scale_factor`, `test_empty_text_fallback`, etc.

**Step 6: Run all tests (including integration)**

```bash
cargo test 2>&1 | tail -20
```

Expected: all tests pass.

**Step 7: Commit**

```bash
git add src/circuit/map.rs
git commit -m "feat: use GloVe-50 word vectors in map_to_algebra_semantic

Replaces character n-gram mapping with GloVe-50 mean vector (85%)
blended with n-gram OOV fallback (15%). Semantically distinct texts
now produce different algebra trajectories.

Fixes the constant-drift finding from the 2026-03-15 empirical run."
```

---

### Task 4: Update the doc comment and README

**Files:**
- Modify: `src/circuit/map.rs` (doc comment at top of file)
- Modify: `README.md`

**Step 1: Update the doc comment on map_to_algebra**

Replace the existing comment block (lines 3–14) with:

```rust
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
```

**Step 2: Update README.md — add a "Finding 2 update" note**

In the README under "Finding 2", add after the existing paragraph:

```markdown
**Update (GloVe-50 bundled embeddings):** The constant-drift finding motivated replacing
character n-grams with a bundled GloVe-50 vocabulary (top-10k words, ~2 MB, embedded at
compile time via `include_bytes!`). The semantic mapping now produces distinct algebra
trajectories for topically different inputs. Re-run with `gravrail proxy` after rebuilding
to verify.
```

**Step 3: Run cargo test one final time**

```bash
cargo test
```

Expected: all tests pass.

**Step 4: Final commit**

```bash
git add src/circuit/map.rs README.md
git commit -m "docs: update map_to_algebra doc comment and README for GloVe embeddings"
```

---

## Summary

| Task | What it delivers |
|------|-----------------|
| 1 | `assets/glove_10k_50d.bin` committed to repo (~2 MB) |
| 2 | `lookup_glove()` + `OnceLock<GloveTable>` in `map.rs` |
| 3 | `map_to_algebra_semantic` uses GloVe mean + n-gram blend |
| 4 | Updated docs |

After completing all tasks, rebuild and re-run the 20-iteration empirical test to confirm that drift values are now varied across different text types rather than constant.

```bash
cargo build --release
# Then re-run whichever empirical test script was used on 2026-03-15
```
