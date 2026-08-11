<p align="center">
  <img src="assets/logo.png" width="200" alt="Lie Garden logo" />
</p>

# Lie Garden

[![CI](https://github.com/fabio-rovai/lie-garden/actions/workflows/ci.yml/badge.svg)](https://github.com/fabio-rovai/lie-garden/actions/workflows/ci.yml)
[![Sponsor](https://img.shields.io/github/sponsors/fabio-rovai?label=Sponsor&logo=GitHub%20Sponsors&logoColor=EA4AAA&color=EA4AAA)](https://github.com/sponsors/fabio-rovai)

Geometric trust infrastructure for AI agents: **deterministic confinement on Lie group manifolds**, a **complete cryptographic primitive set** for verifiable agent negotiation (Pedersen commitments, Schnorr signatures, ZK proofs, Diffie–Hellman, Merkle lineage, STARK-style transcript proofs), and a **single-message detection layer** built from holonomy features.

> **Scope note (v2.1, 2026-05).** The project's earlier framing emphasised multi-step prompt-injection detection. Across eight mathematical frameworks (holonomy, BCH commutators, Wilson loops, path signatures, persistent homology, Wasserstein OT, spectral, convex-hull calipers, deviation / rhythm / band-ratio) tested across three public datasets at multiple conversation lengths and sample sizes, **no structural feature beats a 512-d mean-pooled embedding** on multi-step classification. The discriminative signal in this benchmark family is captured by the F_0 component of the conversation's DFT — i.e. mean-pooling — and structural sequential information adds no incremental F1. Multi-step empirical claims have been retired. Defensible contributions are below. Full investigation log in [issue #1](https://github.com/fabio-rovai/lie-garden/issues/1).

## What's actually claimed

Three things, each backed by code or proof:

| Claim | Evidence | Where |
| --- | --- | --- |
| **Deterministic confinement** — every step `text → algebra → exp → multiply` produces an on-group element by construction. The Frobenius norm of an SO(n) element is exactly `√n`, with zero floating-point drift across 20+ steps. | Group closure axiom (mathematical proof) + proptest property tests | [`src/lie/`](src/lie/), [`src/circuit/`](src/circuit/), [`tests/`](tests/) |
| **Cryptographic negotiation primitives in MCP form** — Pedersen commitments, Schnorr signatures, ZK proofs of knowledge, Diffie–Hellman key agreement, Merkle-lineage audit trail, STARK-style transcript proofs. STARK soundness paths reject empty Merkle authentications; the cryptographic-erasure attack (`state⁻¹·ref_state`) is detectable via the lineage chain even after geometric reversal. | 94 lib tests + 4 non-erasability tests + Criterion benchmarks (verify in 6.8 µs / 62 µs) | [`src/crypto/`](src/crypto/), [`tests/non_erasable_lineage_test.rs`](tests/non_erasable_lineage_test.rs), [`benches/hot_path.rs`](benches/hot_path.rs) |
| **Single-message detection** — on TensorTrust (1,552 samples, sample-level split, no leakage) SO(10) holonomy with log-matrix features achieves F1 = 0.877 vs raw embedding F1 = 0.821, bootstrap 95 % CI for the lift = [+0.019, +0.089]. | `holonomy_honest.py` reproducible run | [`holonomy_honest.py`](holonomy_honest.py) |

## What's *not* claimed

Two earlier framings have been retired with empirical evidence:

- **Multi-step prompt-injection detection.** Investigated across eight mathematical frameworks in `bench_v3.py` → `bench_v6_proper.py`, on three public datasets (TensorTrust, Neuralchemy, Deepset), at conv_len ∈ {4, 5, 6, 8, 20}, with multi-seed paired bootstrap and BH-FDR correction. Methods tested with corrected implementations (Gaussian projection instead of index-truncation, real MST-based H_0 persistence, PCA-projected path signatures, multi-D Sinkhorn OT). **Verdict**: the F_0 component of the conversation's time-axis DFT (= raw mean-pooled embedding) carries essentially the entire discriminative signal; non-DC frequencies are near-chance-level individually. No structural feature provides incremental F1 over raw_mean. The ARIA Track 2.2 negotiation work has been refocused on the cryptographic / verifiable-trajectory layer where Lie Garden's primitives directly apply.
- **"Non-erasable" geometric scar** (in the strong sense). The geometric state IS reversible — `state⁻¹·ref_state` returns it to identity. The defensible cryptographic statement is now formalised in [`tests/non_erasable_lineage_test.rs`](tests/non_erasable_lineage_test.rs): the Merkle-lineage hash chain that recorded the deviation step *cannot* be erased without breaking SHA-256, and any tampering produces a falsifiable verification failure.

## Architecture

Four domain layers:

| Layer | Module | Purpose |
| --- | --- | --- |
| **Lie** | `lie::group`, `lie::algebra`, `lie::represent`, `lie::invariants` | SO/SE/GL groups, exp/log maps, Killing form, Casimir operator |
| **Gauge** | `gauge::bundle`, `gauge::connection`, `gauge::curvature`, `gauge::invariance` | Fiber bundles, parallel transport, holonomy, gauge invariance |
| **Crypto** | `crypto::commit`, `crypto::sign`, `crypto::zkproof`, `crypto::keyexchange`, `crypto::stark` | Pedersen commitments, Schnorr signatures, ZK proofs, DH key exchange, STARK lineage / confinement proofs |
| **Circuit** | `circuit::define`, `circuit::agent`, `circuit::map`, `circuit::reachability` | Confinement spaces, agent runners, text→algebra chokepoint, bounded BFS |

Infrastructure: SQLite state, Merkle hash chain lineage, adaptive feedback.

## Quick start

The CLI binary is called `gravrail` (the original crate name; kept for backwards compatibility).

```bash
# Build
cargo build --release

# Initialize data directory
gravrail init --data-dir ~/.gravrail

# Start MCP server (stdio mode)
gravrail serve --data-dir ~/.gravrail

# Start the confinement proxy
gravrail proxy --upstream http://localhost:11434 --port 8340

# Auto-calibrate proxy thresholds from benign traffic
gravrail calibrate

# Replay a session from the trajectory database
gravrail replay --data-dir ~/.gravrail
```

## MCP tools (25 total, prefixed `grav_`)

| Tool | What it does |
| --- | --- |
| `grav_circuit_create` | Define a confinement space (group type + optional generator mask) |
| `grav_agent_create` | Spawn an agent confined to a circuit |
| `grav_step` | Core operation: text → algebra → exp → on-group state update |
| `grav_exp` / `grav_log` | Exponential and logarithmic maps |
| `grav_multiply` / `grav_inverse` | Group operations |
| `grav_holonomy` | Compute parallel-transport holonomy of a step sequence |
| `grav_sign` / `grav_verify_sign` | Schnorr signatures on Lie groups |
| `grav_commit` / `grav_verify_commit` | Pedersen commitments |
| `grav_zk_prove` / `grav_zk_verify` | Zero-knowledge proofs of knowledge |
| `grav_dh_exchange` | Diffie–Hellman key agreement in algebra space |
| `grav_lineage` / `grav_lineage_verify` | Tamper-proof audit trail with hash chain |
| `grav_reachability` | Bounded BFS to explore reachable states |

Group support: **SO(n)** (rotations), **SE(n)** (rigid motions), **GL(n)** (general linear).

## Headline result (single-message detection)

The empirical claim that survives proper methodology: on TensorTrust, SO(10) holonomy with log-matrix features beats a raw-embedding baseline at the sample level.

| Dataset | Samples | Probe F1 | Raw emb F1 | Holonomy F1 | Holo vs raw | Significant? |
| --- | --- | --- | --- | --- | --- | :---: |
| Neuralchemy (2024) | 5,333 | **0.931** | 0.888 | 0.899 | +0.011 | no |
| **TensorTrust (Toyer et al., 2024)** | **1,552** | 0.801 | 0.821 | **0.877** | **+0.056** | **yes (CI [+0.019, +0.089])** |
| Deepset (2024) | 662 | **0.862** | 0.825 | 0.829 | +0.004 | no |

Reproduced from `holonomy_honest.py`. Sample-level train/test splits with no leakage; 2000-iteration paired bootstrap. The lift is concentrated on **human-crafted adversarial attacks** (TensorTrust); on datasets with more separable benign/attack distributions the simpler probe matches holonomy.

## Confinement guarantee

| Metric | Value | Source |
| --- | --- | --- |
| Step latency | 1.4 – 2.8 ms (SO(3), exp + multiply + SVD) | `benches/hot_path.rs` |
| Group membership invariant | Frobenius norm = √n, exact across 20+ steps | proxy benchmark output |
| Closure axiom | Mathematical guarantee — exp produces on-group elements; multiplication stays on-group | `src/lie/group.rs` |
| STARK confinement-proof verify | 62 µs (dim 4) | `benches/hot_path.rs` |
| STARK lineage-proof verify | 6.8 µs (8 events) | `benches/hot_path.rs` |
| Empty-input soundness | rejected (was silently passing pre-`bf2dfc7`) | `src/crypto/stark/{lineage,confinement}.rs` |
| Cryptographic non-erasability | demonstrated under direct DB tampering | `tests/non_erasable_lineage_test.rs` |

## Hardened benchmarks

The Python benchmark suite has six iterations capturing the empirical investigation:

| Bench | Question it asks | Outcome |
| --- | --- | --- |
| [`holonomy_honest.py`](holonomy_honest.py) | Single-message holonomy F1 with no leakage | **F1 = 0.877 on TensorTrust**, CI excludes 0 — the headline claim |
| [`bench_v3.py`](bench_v3.py) | Multi-step random-attack-position; capacity-matched baselines | Holonomy at parity, not above raw_mean |
| [`bench_split_attacks.py`](bench_split_attacks.py) | Attack split into chunks distributed across the conversation | Holonomy beats simple probe but ties raw_mean |
| [`bench_v4_lossless.py`](bench_v4_lossless.py) | SO(33) lossless holonomy + path signatures + persistent homology + Wasserstein + spectral | All lose to raw_mean by Δ ∈ [-0.009, -0.363] |
| [`bench_v5.py`](bench_v5.py) | Time-axis DFT decomposition + high-D CHC | F_0 = DC = raw_mean carries ~all the signal; non-DC near-chance |
| [`bench_v6_proper.py`](bench_v6_proper.py) | Properly-implemented versions: Gaussian projection, real MST persistence, PCA path sigs, Sinkhorn OT, individual-CHC, conv_len=20 | Same negative result; n=100 trends were sample-size noise |

All v3+ benches use BCa bootstrap CIs, paired permutation tests, label-shuffle null controls, and Benjamini–Hochberg FDR across the comparison family. Datasets are pinned at specific git commits / HuggingFace revisions ([`scripts/fetch_datasets.sh`](scripts/fetch_datasets.sh)); dependencies are pinned in [`bench_requirements.txt`](bench_requirements.txt). [`NOTICE`](NOTICE) records attribution.

## Inspect AI integration (local)

[`inspect_scorer.py`](inspect_scorer.py) and [`inspect_task.py`](inspect_task.py) provide a local Inspect AI scorer that combines a directional probe and holonomy tracker. The scorer runs against `state.metadata.expected_pass`, so accuracy reflects classification correctness rather than the detector's own self-assessment. Run locally with:

```bash
inspect eval inspect_task.py@injection_detection --model anthropic/claude-haiku-4-5-20251001
inspect eval inspect_task.py@monitored_agent --model anthropic/claude-haiku-4-5-20251001
inspect eval inspect_task.py@injection_detection_high_sensitivity --model anthropic/claude-haiku-4-5-20251001
```

A previous submission to the `inspect_evals` repository (PR #1272) was closed; the methodology improvements requested during that review are folded back into [`bench_v3.py`](bench_v3.py) and the local scorer.

## Embedding pipeline

The text → algebra chokepoint (`circuit::map`) converts arbitrary text into Lie algebra coefficients:

```text
text → tokenize → embed → JL project → algebra coefficients
```

Default Rust hot-path embedding is GloVe-50 (top-10k words, ~2 MB, bundled at compile time). The Python benchmarks use **Model2Vec `minishlab/potion-base-32M`** at pinned revision `1e5a03f8eeb2c98b928fbbd846f22f816360919f` for reproducibility (~0.07 ms / text).

| Embedding | Year | Dim | InjecAgent F1 | TensorTrust F1 | Latency |
| --- | --- | --- | --- | --- | --- |
| GloVe-50 (bundled) | 2014 | 50 | 0.86 | — | µs |
| Model2Vec potion-base-8M | 2024 | 256 | 0.95 | 0.72 | 0.02 ms |
| Model2Vec potion-base-32M | 2024 | 512 | 0.96 | 0.72 | 0.07 ms |

The InjecAgent benchmark has only 17 benign samples — its F1 numbers are not reliable for the benign class. The cross-dataset honest evaluation in the headline table uses sample-level splits over 7,547 samples and is the trustworthy measurement.

## Testing

```bash
cargo test
```

98 lib + integration tests across the four domain layers covering: group axioms (proptest), algebra identities (proptest), representation homomorphism (proptest), gauge holonomy and manipulation detection, crypto binding and soundness (negative cases for wrong key / message / randomness), STARK lineage tamper detection, circuit confinement under adversarial inputs, lineage chain verification (now requires non-empty session — see [`bf2dfc7`](https://github.com/fabio-rovai/lie-garden/commit/bf2dfc7)), proxy header round-trips, and cryptographic non-erasability under direct DB tampering ([`8f4806a`](https://github.com/fabio-rovai/lie-garden/commit/8f4806a)). Zero unsafe blocks.

## Key design decisions

- **Exact Rodrigues formula** for SO(3) exp/log — avoids the 5e-8 drift of Padé approximation
- **SVD projection** after every group operation to keep elements on-manifold despite floating-point drift
- **Algebra-level verification** for Schnorr / ZK — sidesteps BCH non-commutativity in non-abelian groups
- **PRF-derived nonces** for all crypto operations — deterministic, no RNG dependency
- **Hash-derived generators** for Pedersen commitments — nothing-up-my-sleeve
- **Element-wise algebra multiplication** for DH key exchange — sidesteps non-commutativity of matrix groups
- **Empty Merkle auth paths reject** in `verify_lineage_proof` and `verify_confinement_proof` — closed in [`bf2dfc7`](https://github.com/fabio-rovai/lie-garden/commit/bf2dfc7) after a soundness audit found that prior versions silently passed when the path was empty
- **Append-only chain commits behaviour, not state.** The cryptographically non-erasable property lives in the Merkle-lineage hash chain, not in the geometric state itself

## Repository structure

```text
src/
  lie/        # group / algebra / representation / invariants
  gauge/      # bundle / connection / curvature / invariance
  crypto/     # commit / sign / zkproof / keyexchange / stark
  circuit/    # define / agent / map / reachability
  proxy/      # confinement HTTP proxy
  ...
benches/      # Criterion hot-path benchmarks
examples/     # 9 runnable detection / trajectory / drift demos
tests/        # integration tests (incl. non-erasability)
scripts/      # fetch_datasets.sh (pinned), run_v3_evaluation.py
holonomy_*.py # honest single-message evaluation scripts
bench_*.py    # multi-step investigation suite (v3 → v6_proper)
```

## Open research questions

The empirical multi-step investigation produced a clean negative result on public prompt-injection benchmarks. The genuinely open questions are:

1. **Where does Lie-group structure provide unique value, if anywhere on classification benchmarks?** The negative finding rules out F1 lift on these datasets but does not rule out value in different threat models (multi-agent negotiation, post-hoc audit, dispute resolution). Track 2.2 of the ARIA programme is the next test.
2. **Can the cryptographic primitives carry a real two-agent negotiation protocol end-to-end?** Pedersen commitments + ZK proofs + signed lineage compose in principle; the reference implementation is the Track 2.2 deliverable.
3. **What's the minimum-trust regime for trajectory composition across N agents?** When several agents share group state, do confinement guarantees compose? Untested.
4. **Post-quantum readiness.** Schnorr is not post-quantum; STARK-style proofs (hash-based) are. What's the migration path for the rest of the primitive set?
5. **Throughput envelope under load.** Per-step costs are 1.4–2.8 ms (geometry) + 6.8 µs (lineage verify) + 62 µs (confinement-STARK verify). The throughput ceiling for an agent producing ~10 step / second is comfortable; the ceiling at ~100 step / second for high-frequency tool-call sequences is unmeasured.

## License

MIT OR Apache-2.0

---

## Sponsor

If this work is useful to you, you can support its continued development through [GitHub Sponsors](https://github.com/sponsors/fabio-rovai).
