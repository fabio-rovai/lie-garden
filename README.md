<p align="center">
  <img src="assets/logo.png" width="200" alt="Lie Garden logo" />
</p>

# Lie Garden

[![CI](https://github.com/fabio-rovai/lie-garden/actions/workflows/ci.yml/badge.svg)](https://github.com/fabio-rovai/lie-garden/actions/workflows/ci.yml)

Geometric trust infrastructure for AI agents: **deterministic confinement on Lie group manifolds**, a **complete cryptographic primitive set** for verifiable agent negotiation (Pedersen commitments, Schnorr signatures, ZK proofs, Diffie–Hellman, Merkle lineage), and a **single-message detection layer** built from holonomy features.

> **Scope note (v2, 2026-05).** The project's earlier framing emphasised multi-step prompt-injection detection. Under proper methodology (`bench_v3`) that empirical claim does not hold against capacity-matched neural baselines. The defensible scope is now (1) the mathematical confinement guarantee, (2) the cryptographic negotiation primitives, and (3) single-message detection on adversarial datasets. See [issue #1](https://github.com/fabio-rovai/lie-garden/issues/1) for the full record.

## What's actually claimed

Three things, each backed by code or proof:

| Claim | Evidence | Where |
| --- | --- | --- |
| **Deterministic confinement** — every step `text → algebra → exp → multiply` produces an on-group element by construction. The Frobenius norm of an SO(n) element is exactly `√n`, with zero floating-point drift across 20+ steps. | Group closure axiom (mathematical proof) + proptest property tests | [`src/lie/`](src/lie/), [`src/circuit/`](src/circuit/), [`tests/`](tests/) |
| **Cryptographic negotiation primitives in MCP form** — Pedersen commitments, Schnorr signatures, ZK proofs of knowledge, Diffie–Hellman key agreement, Merkle-lineage audit trail. STARK soundness paths reject empty Merkle authentications. | 94 lib tests + Criterion benchmarks (verify in 6.8 µs / 62 µs) | [`src/crypto/`](src/crypto/), [`benches/hot_path.rs`](benches/hot_path.rs) |
| **Single-message detection** — on TensorTrust (1,552 samples, sample-level split, no leakage) SO(10) holonomy with log-matrix features achieves F1 = 0.877 vs raw embedding F1 = 0.821, bootstrap 95 % CI for the lift = [+0.019, +0.089]. | `holonomy_honest.py` reproducible run | [`holonomy_honest.py`](holonomy_honest.py) |

## What's *not* claimed

Two earlier framings have been retired:

- **Multi-step prompt-injection detection.** Tested under `bench_v3` (text-level dedup, randomised attack position, capacity-matched baselines, BCa bootstrap, BH-FDR) on three datasets across five seeds. Holonomy did not lift F1 over a simple linear probe, and a follow-up split-attack benchmark showed holonomy is parity with — not better than — a 512-dim raw mean embedding baseline. Multi-step claims are paused until a regime is found where holonomy specifically beats capacity-matched neural baselines.
- **"Non-erasable" geometric scar.** [`holonomy_honest.py:591`](holonomy_honest.py) explicitly lists this under "WHAT WE CANNOT CLAIM". The defensible statement is "persistent residue under non-commutative composition" — harder to fully erase without knowledge of the calibrated reference state, but not mathematically irreversible.

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

## Hardened benchmarks

Under [`scripts/`](scripts/) and the top-level Python files:

- **`bench_v3.py`** — multi-step random-attack-position benchmark with text-level dedup, capacity-matched baselines (probe / raw_mean / random_proj / combined), BCa bootstrap, permutation test, label-shuffle null control, every RNG threaded through `--seed`. Drives [`scripts/run_v3_evaluation.py`](scripts/run_v3_evaluation.py) which adds Benjamini–Hochberg FDR across the comparison family.
- **`bench_split_attacks.py`** — splits each attack at sentence boundaries and distributes chunks across positions of a multi-message conversation (the regime where holonomy is *theoretically* supposed to help). Inherits all v3 hardening.
- **`holonomy_honest.py`** — the reproducible script behind the headline TensorTrust result above. Includes a per-call `safe_logm` fallback counter that aborts the run if more than 5 % of calls hit the zero-feature fallback.
- **`scripts/fetch_datasets.sh`** — pulls TensorTrust, InjecAgent, Deepset, and Neuralchemy at **pinned git commits and HuggingFace revisions**. Combined with [`bench_requirements.txt`](bench_requirements.txt) and the pinned model revision in `bench_v3.py`, results in `v*_results*.json` are reproducible end-to-end.

[NOTICE](NOTICE) records attribution for all four datasets and the embedding model.

## Inspect AI integration (local)

[`inspect_scorer.py`](inspect_scorer.py) and [`inspect_task.py`](inspect_task.py) provide a local Inspect AI scorer that combines a directional probe and holonomy tracker. The scorer runs against `state.metadata.expected_pass`, so accuracy reflects classification correctness rather than the detector's own self-assessment. Run locally with:

```bash
inspect eval inspect_task.py@injection_detection --model anthropic/claude-haiku-4-5-20251001
inspect eval inspect_task.py@monitored_agent --model anthropic/claude-haiku-4-5-20251001
inspect eval inspect_task.py@injection_detection_high_sensitivity --model anthropic/claude-haiku-4-5-20251001
```

A previous submission of these tasks to the `inspect_evals` repository (PR #1272) was closed; the methodology improvements requested during that review have since been folded back into `bench_v3` and into the local scorer (e.g. labelling against `expected_pass` rather than the detector's own verdict).

## Embedding pipeline

The text → algebra chokepoint (`circuit::map`) converts arbitrary text into Lie algebra coefficients:

```
text → tokenize → embed → JL project → algebra coefficients
```

Default Rust hot-path embedding is GloVe-50 (top-10k words, ~2 MB, bundled at compile time). The Python benchmarks use **Model2Vec `minishlab/potion-base-32M`** at pinned revision `1e5a03f8eeb2c98b928fbbd846f22f816360919f` for reproducibility (~0.07 ms / text). The geometric layer is embedding-agnostic; `map_to_algebra` runs once per turn, so transformer-based embeddings add negligible latency relative to LLM generation.

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

94 lib tests across the four domain layers covering: group axioms (proptest), algebra identities (proptest), representation homomorphism (proptest), gauge holonomy and manipulation detection, crypto binding and soundness (negative cases for wrong key / message / randomness), STARK lineage tamper detection, circuit confinement under adversarial inputs, lineage chain verification (now requires non-empty session — see [`bf2dfc7`](https://github.com/fabio-rovai/lie-garden/commit/bf2dfc7)), and proxy header round-trips. Zero unsafe blocks.

## Key design decisions

- **Exact Rodrigues formula** for SO(3) exp/log — avoids the 5e-8 drift of Padé approximation
- **SVD projection** after every group operation to keep elements on-manifold despite floating-point drift
- **Algebra-level verification** for Schnorr / ZK — sidesteps BCH non-commutativity in non-abelian groups
- **PRF-derived nonces** for all crypto operations — deterministic, no RNG dependency
- **Hash-derived generators** for Pedersen commitments — nothing-up-my-sleeve
- **Element-wise algebra multiplication** for DH key exchange — sidesteps non-commutativity of matrix groups
- **Empty Merkle auth paths reject** in `verify_lineage_proof` and `verify_confinement_proof` — closed in [`bf2dfc7`](https://github.com/fabio-rovai/lie-garden/commit/bf2dfc7) after a soundness audit found that prior versions silently passed when the path was empty

## Repository structure

```
src/
  lie/        # group / algebra / representation / invariants
  gauge/      # bundle / connection / curvature / invariance
  crypto/     # commit / sign / zkproof / keyexchange / stark
  circuit/    # define / agent / map / reachability
  proxy/      # confinement HTTP proxy
  ...
benches/      # Criterion hot-path benchmarks
examples/     # 9 runnable detection / trajectory / drift demos
tests/        # integration tests
scripts/      # fetch_datasets.sh, run_v3_evaluation.py
```

## Open research questions

1. **Where (if anywhere) does holonomy specifically beat capacity-matched neural baselines?** `bench_v3` and `bench_split_attacks` rule out random-attack-position and naïvely-split-attack regimes on three datasets. Untested: adversarially-chunked attacks, very long trajectories, learned classifiers over per-step algebra states (rather than just the final accumulated state).
2. **Optimal group selection.** Which Lie group family (SO / SE / GL) and dimension maximises detection power per unit compute? Single-message holonomy lift on TensorTrust grows from SO(5) (no lift) through SO(10) (+0.056 F1, significant) to SO(25) (uncertain — needs reruns under v3 methodology).
3. **Adaptive thresholds.** Can holonomy thresholds be auto-calibrated from benign traffic without labelled harmful examples? `gravrail calibrate` does this for proxy-side norms; the geometric thresholds are still set manually.
4. **Multi-agent topology.** How do confinement guarantees compose when multiple agents interact through shared group state? This is the Track 2.2 "Negotiation" question.
5. **STARK proof overhead in the agent loop.** Per-step ZK proofs for tamper-evident audit cost 62 µs for confinement and 6.8 µs for lineage today — what's the throughput ceiling for a real agent that produces ~10 step / second?

## License

MIT OR Apache-2.0
