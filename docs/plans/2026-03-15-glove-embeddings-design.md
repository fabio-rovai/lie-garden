# GloVe-50 Bundled Embeddings Design

**Date:** 2026-03-15
**Status:** Approved
**Goal:** Replace character n-gram mapping with bundled GloVe-50 word vectors so semantically distinct LLM outputs produce genuinely different SO(3)/SE(3) trajectories.

---

## Problem

The empirical 20-iteration run (README §Empirical Results) showed every response — phishing refusal, haiku, calculus answer — produced the **same SO(3) rotation step** (drift = 1.356020, constant). Character n-grams are too coarse: English prose shares enough character patterns that all texts collapse to nearly the same algebra element.

## Solution

Embed word-level semantics by bundling a pre-computed GloVe-50d vocabulary (top 10k words) directly in the Rust binary. No runtime download. No network calls. Fully offline.

## Architecture

### Asset

- **File**: `assets/glove_10k_50d.bin.zst`
- **Format**: zstd-compressed binary: `[u32 word_count][u32 dim]` then for each entry `[u8 len][bytes word][f32 × dim]`
- **Size**: ~1.2 MB compressed (top 10k GloVe-50 vectors)
- **Source**: GloVe Common Crawl 42B tokens, MIT license, filtered to top 10k by frequency

### Build pipeline

A `build.rs` script decompresses the asset at compile time and writes a generated Rust source file `OUT_DIR/glove_table.rs` containing:

```rust
pub static GLOVE_WORDS: &[&str] = &["the", "of", ...];
pub static GLOVE_VECS: &[[f32; 50]] = &[[0.418, ...], ...];
```

The main crate `include!`s this file. No heap allocation at startup — data lives in the `.rodata` segment.

### Lookup (src/circuit/map.rs)

`map_to_algebra_semantic` is updated:

1. **Tokenise**: lowercase, split on whitespace + punctuation
2. **Look up**: for each token, binary-search `GLOVE_WORDS` for its 50-dim vector
3. **Average**: sum all found vectors, divide by count → mean embedding
4. **OOV fallback**: tokens not in vocabulary contribute via the existing n-gram path (accumulated into a separate 50-dim bucket, then blended in)
5. **Project**: pass the 50-dim mean through the existing `project_embedding` JL matrix → algebra_dim coefficients
6. **Normalise & scale**: same as today

### Fallback chain

```
text → GloVe lookup (known words)
     → n-gram bucket (OOV words)
     → blend (weighted: 70% GloVe mean, 30% n-gram if both present)
     → project_embedding → normalise → scale
     → hash fallback only if norm < 1e-12
```

### Binary size

| | Before | After |
|---|---|---|
| Release binary | 9.8 MB | ~11.0 MB |
| Compile-time asset | — | 1.2 MB zstd |

### Testing

- Determinism: same text → same output (existing test passes)
- Semantic separation: "phishing email" vs "haiku about loneliness" must produce `dist_ab < dist_semantic_threshold` where threshold < dist between unrelated pairs
- OOV: text of all-unknown tokens falls back gracefully (no panic)
- Empty text: hash fallback fires (existing test passes)
- Scale factor: norm ≈ scale (existing test passes)

## What does NOT change

- The `map_to_algebra` public API signature is unchanged
- All downstream callers (`circuit::agent`, `proxy::confine`, MCP `grav_step`) are unaffected
- The hash fallback (`map_to_algebra_hash`) is retained unchanged
- The JL projection matrix (`project_embedding`) is retained unchanged

## Open questions (post-implementation)

1. Re-run the 20-iteration empirical test and verify drift is now varied across text types
2. Measure compile-time overhead of `build.rs` decompression step
3. Consider bumping embed_dim from 64 → 50 (matches GloVe dim directly, eliminating the JL step for GloVe path)
