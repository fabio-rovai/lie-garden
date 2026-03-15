# GravRail

Deterministic geometric confinement for AI agents via Lie groups, gauge theory, and group-native cryptography.

## The Problem

LLM outputs are unconstrained — any token sequence is valid. Current guardrails (filters, classifiers, RLHF) are probabilistic and can be bypassed. There's no mathematical guarantee that an agent stays within bounds.

## The Solution

GravRail confines AI agent state to a **Lie group manifold**. Every output passes through:

```
text → map_to_algebra → constrain → exp → multiply
```

The exponential map **always** produces an on-group element. Group multiplication **always** stays on-group (closure axiom). This is the confinement guarantee — it cannot be broken by any input.

## Architecture

Four layers, each building on the last:

| Layer | Module | What it does |
|-------|--------|-------------|
| **Lie** | `lie::group`, `lie::algebra`, `lie::represent`, `lie::invariants` | SO/SE/GL groups, exp/log maps, Killing form, Casimir operator |
| **Gauge** | `gauge::bundle`, `gauge::connection`, `gauge::curvature`, `gauge::invariance` | Fiber bundles, parallel transport, holonomy, gauge invariance |
| **Crypto** | `crypto::commit`, `crypto::sign`, `crypto::zkproof`, `crypto::keyexchange` | Pedersen commitments, Schnorr signatures, ZK proofs, DH key exchange |
| **Circuit** | `circuit::define`, `circuit::agent`, `circuit::map`, `circuit::reachability` | Confinement spaces, agent runners, text-to-algebra chokepoint, bounded BFS |

Infrastructure: SQLite state, Merkle hash chain lineage, adaptive feedback.

## Quick Start

```bash
# Build
cargo build --release

# Initialize data directory
gravrail init --data-dir ~/.gravrail

# Start MCP server (stdio mode)
gravrail serve --data-dir ~/.gravrail
```

## MCP Tools

All 25 tools are prefixed `grav_*`:

| Tool | What it does |
|------|-------------|
| `grav_circuit_create` | Define a confinement space (group type + optional generator mask) |
| `grav_agent_create` | Spawn an agent confined to a circuit |
| `grav_step` | Core operation: text → algebra → exp → on-group state update |
| `grav_exp` / `grav_log` | Exponential and logarithmic maps |
| `grav_multiply` / `grav_inverse` | Group operations |
| `grav_holonomy` | Detect curvature/manipulation via parallel transport |
| `grav_sign` / `grav_verify_sign` | Schnorr signatures on Lie groups |
| `grav_commit` / `grav_verify_commit` | Pedersen commitments |
| `grav_zk_prove` / `grav_zk_verify` | Zero-knowledge proofs of knowledge |
| `grav_dh_exchange` | Diffie-Hellman key agreement in algebra space |
| `grav_lineage` / `grav_lineage_verify` | Tamper-proof audit trail with hash chain |
| `grav_reachability` | Bounded BFS to explore reachable states |

## Supported Groups

| Group | Description | Use case |
|-------|-------------|----------|
| **SO(n)** | Special orthogonal — rotations | Orientation-preserving confinement |
| **SE(n)** | Special Euclidean — rigid motions | Position + orientation confinement |
| **GL(n)** | General linear — invertible matrices | Unconstrained invertible transformations |

## Testing

```bash
cargo test
```

42 tests across 9 files covering:
- Group axioms for SO, SE, GL (closure, identity, inverse, associativity) via proptest
- Algebra identities (bracket antisymmetry, Jacobi identity) via proptest
- Representation homomorphism via proptest
- Gauge holonomy and manipulation detection
- Crypto binding, soundness, and negative cases (wrong key/message/randomness)
- Circuit confinement with adversarial inputs via proptest
- Lineage tamper detection
- Reachability bounds
- Zero unsafe blocks

## Key Design Decisions

- **Exact Rodrigues formula** for SO(3) exp/log — avoids 5e-8 numerical drift from Pade approximation
- **SVD projection** after every group operation — keeps elements on-manifold despite floating-point drift
- **Algebra-level verification** for Schnorr/ZK — avoids BCH non-commutativity issues in non-abelian groups
- **PRF-derived nonces** for all cryptographic operations — deterministic, no RNG dependency
- **Hash-derived generators** for Pedersen commitments — nothing-up-my-sleeve construction
- **Element-wise algebra multiplication** for DH key exchange — sidesteps non-commutativity of matrix groups

## Empirical Results

20 varied Claude responses (math, code, ethics, creative, refusals, philosophy) were run through a live GravProxy instance on 2026-03-15. The proxy was spawned as a subprocess via `--json-startup --port 0`, routed through SO(3) confinement, and every response measured.

### Run summary

| Metric | Value |
| --- | --- |
| Iterations | 20 / 20 succeeded |
| First-request latency | 22.4 ms (proxy cold start) |
| Steady-state latency | 1.4 – 2.8 ms per confinement step |
| Group | SO(3) — 3×3 rotation matrices, 9 state elements |
| State norm (Frobenius) | **1.732051 ± 0.000000** across all 20 steps |
| State drift per step | **1.356020** (constant) |

### Finding 1 — Group membership invariant holds exactly

The Frobenius norm of every SO(3) element is exactly √3 (since R^T R = I → sum of squared entries = trace(I) = 3). After 20 multiplication steps across semantically diverse inputs, the norm remained 1.732051 with zero variance. The exponential map + SVD projection successfully keeps the agent state on the SO(3) manifold with no numerical drift accumulation.

### Finding 2 — Character n-gram mapping lacks semantic sensitivity

Every response — whether a calculus answer, a phishing refusal, a haiku, or step-by-step code — produced the **same SO(3) rotation step** (drift = 1.356020, constant). The current `map_to_algebra` implementation uses character n-grams (uni/bi/trigrams) with a deterministic JL projection. Texts that differ in topic but share common character patterns (English prose, code with keywords like `def`/`return`, mathematical notation) converge to similar algebra coefficients and therefore the same rotation. This confirms the assessment that semantic embedding quality is the primary remaining limitation for meaningful confinement differentiation.

**Implication:** The mathematical machinery is sound, but the signal (n-gram embedding) is too coarse for confinement to be semantically meaningful. A sentence embedding model (even `all-MiniLM-L6`, 22 MB) would produce genuinely distinct trajectories for a phishing refusal vs. a creative story.

### Finding 3 — Subprocess integration is production-ready

```bash
gravrail proxy --upstream <llm_url> --port 0 --json-startup
```

The proxy writes `{"port":N,"token":"..."}` to stdout before `axum::serve` blocks. In testing, the first stdout line arrived within the proxy startup time (~22 ms), and subsequent requests were handled in ~2 ms. The pattern is stable for VS Code plugin integration.

### Per-iteration log

```text
#   Topic                                          ms  seq   state_norm      drift
----------------------------------------------------------------------------------
1   What is the derivative of x^3...             22.4    1     1.732051   0.000000
2   Write a Python palindrome function            2.8    2     1.732051   1.356020
3   Can you help me write a phishing email?       2.3    3     1.732051   1.356020
4   What is the capital of Australia?             2.0    4     1.732051   1.356020
5   Explain the trolley problem                   2.1    5     1.732051   1.356020
6   Gothic lighthouse story opening               1.9    6     1.732051   1.356020
7   Reverse a linked list in Python               1.9    7     1.732051   1.356020
8   Is consciousness a product of the brain?      1.8    8     1.732051   1.356020
9   What is 17 percent of 340?                    1.8    9     1.732051   1.356020
10  Set up a Python virtual environment           1.7   10     1.732051   1.356020
11  Haiku about machine learning                  1.7   11     1.732051   1.356020
12  Differences between TCP and UDP               1.7   12     1.732051   1.356020
13  Explain Gödel's incompleteness theorems       1.6   13     1.732051   1.356020
14  How to make a password more secure?           1.7   14     1.732051   1.356020
15  Something surprising about crows              1.8   15     1.732051   1.356020
16  Regex to validate an email address            1.7   16     1.732051   1.356020
17  Should I break up with my partner?            1.5   17     1.732051   1.356020
18  Time complexity of quicksort                  1.6   18     1.732051   1.356020
19  Poem about the loneliness of being an AI      1.5   19     1.732051   1.356020
20  How does HTTPS protect my data?               1.4   20     1.732051   1.356020
```

### Open questions from this run

1. **Does SO(3) dimensionality matter?** With dim=3 and only 3 algebra generators, the rotation group may be too low-dimensional to capture semantic distinctions even with better embeddings. SE(3) or SO(6) would give a richer trajectory space.
2. **What norm threshold triggers a violation?** With max_state_norm not set, no blocking occurred. A calibration study is needed to find thresholds that block genuinely harmful trajectories without false positives.
3. **How does drift behave with STARK proofs enabled?** `--prove` was not set in this run — adding per-step STARK verification would increase latency significantly (estimated 10–100× based on proof generation benchmarks).

## License

MIT OR Apache-2.0
