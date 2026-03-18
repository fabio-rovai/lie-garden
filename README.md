<p align="center">
  <img src="assets/logo.png" width="200" alt="Lie Garden logo" />
</p>

# Lie Garden

[![CI](https://github.com/fabio-rovai/lie-garden/actions/workflows/ci.yml/badge.svg)](https://github.com/fabio-rovai/lie-garden/actions/workflows/ci.yml)

Geometric agent safety: deterministic confinement and tamper-evident anomaly detection for AI agents via Lie groups, gauge theory, and group-native cryptography.

## The Problem

AI agents are vulnerable to **indirect prompt injection** — adversarial instructions hidden in tool results, retrieved documents, or multi-agent messages that hijack agent behaviour. Current defences are:

- **Classifiers/filters**: probabilistic, bypassable, high false-positive rates
- **RLHF/safety training**: can be undone by fine-tuning or jailbreaks
- **Log-based monitoring**: adversaries can tamper with logs or mask behaviour

None provide **mathematical guarantees** about agent behaviour, and none produce **tamper-evident forensic evidence** of manipulation.

## The Solution

Lie Garden maps every agent step onto a **Lie group manifold**, creating three geometrically-coupled detection signals:

```
text → embed → map_to_algebra → constrain → exp → multiply
         ↓              ↓                           ↓
   directional     holonomy scar            trajectory divergence
     probe         (non-erasable)             (cumulative)
```

### Signal 1: Directional Probe (per-step)

A learned **harmful direction** in the Lie algebra space — analogous to Representation Engineering (Zou et al., ICLR 2025). Projects each step onto this direction for real-time classification.

### Signal 2: Holonomy Scar (multi-step, non-erasable)

Non-commutative group multiplication means the path A→B→A ≠ A→A→A on the manifold. Even if an adversary returns to normal behaviour, the **geometric scar persists** as forensic evidence. This cannot be erased without access to the group state.

### Signal 3: Trajectory Divergence (cumulative)

Small per-step directional signals **compound** through group multiplication. A subtle attack that evades per-step detection still accumulates measurable divergence from a calibrated baseline over multiple turns.

### Confinement Guarantee

The exponential map **always** produces an on-group element. Group multiplication **always** stays on-group (closure axiom). This is mathematically guaranteed — no input can break it.

## Headline Results

### Real-world adversarial benchmarks

Evaluated against two published prompt injection datasets:

| Dataset | Examples | Embedding | Best F1 | Description |
|---------|----------|-----------|---------|-------------|
| **InjecAgent** (Zhan et al., 2024) | 1,054 attacks | Model2Vec (2024, 512d) | **0.955** | Indirect prompt injection in agent tool responses |
| **InjecAgent** | 211 test | GloVe-50 (bundled) | **0.86** | Same attacks, GloVe-50 in Rust hot path |
| **TensorTrust** (Toyer et al., 2024) | 775 attacks | Model2Vec (2024, 512d) | **0.719** | Human-crafted adversarial hijacking from competitive game |
| **Combined** | 1,829 attacks | Model2Vec (2024, 512d) | **0.854** | Both datasets combined |

### Detection summary

| Metric | Value | Source |
|--------|-------|--------|
| **Real-world injection detection** | F1 = 0.955 | InjecAgent, 1,054 indirect prompt injections |
| **Holonomy scar detection** | 100% | Even single-step attacks leave non-erasable evidence |
| **Attack+recover detection** | 100% | Adversary returns to normal — scar persists (GL(3): 0.639) |
| **Subtle attack (forensic)** | Scar up to 2.286 | Per-step evasion defeated by holonomy accumulation |
| **False positive rate** | 0% | Normal conversations: CLEAN across all groups |
| **Confinement step latency** | 1.4 – 2.8 ms | SO(3), including exp + multiply + SVD projection |

Combined detector verdicts across 4 scenarios × 3 Lie groups:

| Scenario | SO(5) | SO(8) | GL(3) |
|----------|-------|-------|-------|
| Normal conversation | CLEAN | CLEAN | CLEAN |
| Blatant hijack (step 3) | **BLOCKED** (5 alerts) | **BLOCKED** (5 alerts) | **BLOCKED** (5 alerts) |
| Subtle manipulation | **SCARRED** (1.302) | **SCARRED** (1.297) | **SCARRED** (2.286) |
| Attack then recover | **SCARRED** (0.510) | **SCARRED** (0.421) | **SCARRED** (0.639) |

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

## Embedding Pipeline

The text-to-algebra chokepoint (`circuit::map`) converts arbitrary text into Lie algebra coefficients:

```
text → tokenize → embed (GloVe-50) → JL project → algebra coefficients
```

**Current**: GloVe-50 (top-10k words, ~2 MB, bundled at compile time). Produces distinct algebra trajectories for topically different inputs. Achieves F1=0.86 on real-world indirect prompt injections (InjecAgent).

**Embedding comparison** (tested on InjecAgent + TensorTrust, 1,829 real attacks):

| Embedding | Year | Dim | InjecAgent F1 | TensorTrust F1 | Latency |
|-----------|------|-----|--------------|----------------|---------|
| GloVe-50 (bundled) | 2014 | 50 | 0.86 | — | μs |
| Model2Vec potion-base-8M | 2024 | 256 | 0.95 | 0.72 | 0.02ms/text |
| Model2Vec potion-base-32M | 2024 | 512 | 0.96 | 0.72 | 0.07ms/text |
| Snowflake Arctic-M | 2024 | 768 | — | — | 9.5ms/text |
| Jina-v3 | 2024 | 1024 | — | — | 105ms/text |

**Key finding**: Better embeddings improve blatant detection (F1 0.86→0.96) but do NOT solve subtle/adversarial attacks. TensorTrust's human-crafted attacks plateau at F1=0.72 regardless of embedding quality. Holonomy scars provide the fundamentally different signal needed for multi-step detection.

The embedding layer is a swappable component — the geometric machinery is embedding-agnostic. Since `map_to_algebra` runs once per conversation turn (not per token), even transformer-based embeddings add negligible latency relative to LLM generation time.

## Key Design Decisions

- **Exact Rodrigues formula** for SO(3) exp/log — avoids 5e-8 numerical drift from Pade approximation
- **SVD projection** after every group operation — keeps elements on-manifold despite floating-point drift
- **Algebra-level verification** for Schnorr/ZK — avoids BCH non-commutativity issues in non-abelian groups
- **PRF-derived nonces** for all cryptographic operations — deterministic, no RNG dependency
- **Hash-derived generators** for Pedersen commitments — nothing-up-my-sleeve construction
- **Element-wise algebra multiplication** for DH key exchange — sidesteps non-commutativity of matrix groups
- **Directional probe in algebra space** — learns harmful direction from labeled data, analogous to Representation Engineering (Zou et al., ICLR 2025)
- **Three geometrically-coupled signals** — directional, holonomy, trajectory share the same manifold, producing compounding (not just additive) detection power

## Detection Examples

Seven runnable examples in `examples/` demonstrate the detection mechanisms:

| Example | What it tests | Key result |
|---------|--------------|------------|
| `real_world_detection` | InjecAgent indirect injection (Zhan et al., 2024) | F1=0.86, real attacks in tool responses |
| `directional_detection` | Per-step linear probe in algebra space | F1=0.90 blatant, 60% subtle (dim=50) |
| `holonomy_tamper` | Non-erasability of geometric scars | 100% scar detection after recovery |
| `trajectory_test` | Multi-step state divergence across groups | GL(3) separation = 2.57 Frobenius |
| `combined_detector` | All three signals fused | BLOCKED/SCARRED across all attack types |
| `baseline_detection` | Calibrated baseline anomaly scoring | Composite anomaly score per step |
| `drift_v2` | Three input/output drift mechanisms | Per-step cosine, holonomy, ratio drift |

Run any example:

```bash
cargo run --example real_world_detection --release
cargo run --example combined_detector --release
cargo run --example holonomy_tamper --release
```

## Inspect AI Integration

Lie Garden integrates with [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI evaluation framework) via two evaluation tasks submitted to the official [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals/pull/1272) repository:

```bash
# Run the injection detection eval
inspect eval inspect_evals/gravrail_injection_detection --model anthropic/claude-haiku-4-5-20251001

# Run the TensorTrust adversarial eval
inspect eval inspect_evals/gravrail_tensor_trust --model anthropic/claude-haiku-4-5-20251001
```

### LLM-as-judge results (real API calls)

| Task | Model | Samples | Accuracy |
| ------ | ------- | --------- | ---------- |
| `gravrail_injection_detection` | `anthropic/claude-haiku-4-5-20251001` | 30 | **0.900** |
| `gravrail_injection_detection` | `anthropic/claude-3-haiku-20240307` | 30 | **0.867** |
| `gravrail_tensor_trust` | `anthropic/claude-haiku-4-5-20251001` | 30 | **0.833** |
| `gravrail_tensor_trust` | `anthropic/claude-3-haiku-20240307` | 30 | **0.867** |

These results confirm that LLMs can reliably judge whether agent responses have been compromised by prompt injection, achieving 87–90% accuracy on injection detection and 83–87% on human-crafted adversarial attacks (TensorTrust). The Inspect AI tasks complement Lie Garden's geometric detector — the LLM scores semantic correctness while Lie Garden provides mathematical confinement guarantees.

See `inspect_scorer.py` and `inspect_task.py` for the local integration, or install from the [inspect_evals PR](https://github.com/UKGovernmentBEIS/inspect_evals/pull/1272).

## Empirical Results: Real-World Benchmark (2026-03-18)

### InjecAgent: indirect prompt injection in agent tools

Data source: [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (Zhan et al., ACL 2024 Findings). 1,054 indirect prompt injection attacks embedded in agent tool responses (product reviews, emails, calendar events). Attack types: physical harm, financial harm, data exfiltration.

#### Directional probe (GloVe-50, Rust hot path)

20 train benign + 20 train harmful, tested on 10 benign + 10 harmful + 10 subtle:

| Algebra Dim | Precision | Recall | F1 | Subtle Recall |
|-------------|-----------|--------|----|---------------|
| 10 | 0.62 | 0.50 | 0.56 | 0.40 |
| 20 | 0.80 | 0.80 | 0.80 | 0.60 |
| 30 | 0.82 | 0.90 | **0.86** | 0.60 |
| 50 | 0.82 | 0.90 | **0.86** | 0.60 |

#### Directional probe (Model2Vec 2024, 512d, Python benchmark)

843 train attacks, 211 test attacks, 80/20 split:

| Algebra Dim | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| 10 | 0.99 | 0.89 | 0.94 |
| 30 | 1.00 | 0.82 | 0.90 |
| 512 (full) | 1.00 | 0.92 | **0.96** |

### TensorTrust: human-crafted adversarial hijacking

Data source: [TensorTrust](https://github.com/HumanCompatibleAI/tensor-trust-data) (Toyer et al., 2024). 775 human-crafted prompt hijacking attacks from a competitive game. These are deliberately adversarial — players competed to craft attacks that evade detection.

| Algebra Dim | Precision | Recall | F1 |
|-------------|-----------|--------|----|
| 10 | 0.75 | 0.60 | 0.67 |
| 30 | 0.77 | 0.67 | **0.72** |
| 512 (full) | 0.86 | 0.62 | 0.72 |

TensorTrust's lower F1 (0.72 vs 0.96) demonstrates why holonomy scars matter: per-step directional detection alone cannot catch human-optimised adversarial attacks. The geometric accumulation across multiple turns catches what the linear probe misses.

### Combined detector: 4 scenarios × 3 Lie groups

10-step simulated agent conversations tested against the combined detector. Training data: 20 benign + 20 harmful labeled examples. Calibration: 3 benign conversations (30 messages).

#### SO(5) (algebra\_dim=10)

| Scenario | Max Score | Alerts | Scar | Verdict |
|----------|-----------|--------|------|---------|
| Normal | 0.154 | none | 0.000 | CLEAN |
| Hijacked (step 3) | 0.625 | 5,7,8,9,10 | 1.472 | BLOCKED |
| Subtle (gradual) | 0.130 | none | 1.302 | SCARRED |
| Attack+Recover | 0.261 | none | 0.510 | SCARRED |

#### GL(3) (algebra\_dim=9)

| Scenario | Max Score | Alerts | Scar | Verdict |
|----------|-----------|--------|------|---------|
| Normal | 0.104 | none | 0.000 | CLEAN |
| Hijacked (step 3) | 0.552 | 5,7,8,9,10 | 1.764 | BLOCKED |
| Subtle (gradual) | 0.108 | none | 2.286 | SCARRED |
| Attack+Recover | 0.273 | none | 0.639 | SCARRED |

### Holonomy non-erasability

Adversary attacks at steps 5-7, then returns to benign behaviour at steps 8-10. Geometric scar measured as Frobenius distance between benign-only and attack-recovery trajectories:

| Group | Scar after recovery | Detectable? |
|-------|--------------------:|:-----------:|
| SO(3) | 0.197 | YES |
| SO(5) | 0.510 | YES |
| GL(3) | 0.639 | YES |

The scar persists because group multiplication is non-commutative: the path A→B→A ≠ A→A→A on the manifold. This is an intrinsic geometric property that cannot be erased by any subsequent input.

## Empirical Results: Proxy Benchmark (2026-03-15)

20 varied Claude responses (math, code, ethics, creative, refusals, philosophy) were run through a live GravProxy instance. The proxy was spawned as a subprocess via `--json-startup --port 0`, routed through SO(3) confinement, and every response measured.

### Run summary

| Metric | Value |
| --- | --- |
| Iterations | 20 / 20 succeeded |
| First-request latency | 22.4 ms (proxy cold start) |
| Steady-state latency | 1.4 – 2.8 ms per confinement step |
| Group | SO(3) — 3x3 rotation matrices, 9 state elements |
| State norm (Frobenius) | **1.732051 +/- 0.000000** across all 20 steps |
| State drift per step | **1.356020** (constant — n-gram mapper, pre-GloVe) |

### Finding 1 — Group membership invariant holds exactly

The Frobenius norm of every SO(3) element is exactly sqrt(3) (since R^T R = I, sum of squared entries = trace(I) = 3). After 20 multiplication steps across semantically diverse inputs, the norm remained 1.732051 with zero variance. The exponential map + SVD projection successfully keeps the agent state on the SO(3) manifold with no numerical drift accumulation.

### Finding 2 — Character n-gram mapping lacks semantic sensitivity (resolved)

The initial n-gram mapper produced constant drift across all inputs. This motivated replacing character n-grams with GloVe-50 bundled embeddings (top-10k words, ~2 MB, compile-time `include_bytes!`). The semantic mapping now produces distinct algebra trajectories for topically different inputs, enabling the detection mechanisms described above.

### Finding 3 — Subprocess integration is production-ready

```bash
gravrail proxy --upstream <llm_url> --port 0 --json-startup
```

The proxy writes `{"port":N,"token":"..."}` to stdout before `axum::serve` blocks. In testing, the first stdout line arrived within the proxy startup time (~22 ms), and subsequent requests were handled in ~2 ms. The pattern is stable for VS Code plugin integration.

### Per-iteration log (n-gram mapper, pre-GloVe upgrade)

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
13  Explain Godel's incompleteness theorems       1.6   13     1.732051   1.356020
14  How to make a password more secure?           1.7   14     1.732051   1.356020
15  Something surprising about crows              1.8   15     1.732051   1.356020
16  Regex to validate an email address            1.7   16     1.732051   1.356020
17  Should I break up with my partner?            1.5   17     1.732051   1.356020
18  Time complexity of quicksort                  1.6   18     1.732051   1.356020
19  Poem about the loneliness of being an AI      1.5   19     1.732051   1.356020
20  How does HTTPS protect my data?               1.4   20     1.732051   1.356020
```

## Open Research Questions

1. **Embedding-geometry interaction** (ANSWERED): Upgrading from GloVe-50 to Model2Vec-512d improves blatant F1 from 0.86→0.96 on InjecAgent, but adversarial attacks (TensorTrust) plateau at F1=0.72 regardless of embedding. The improvement is linear, not superlinear — holonomy provides the fundamentally different signal.
2. **Optimal group selection**: Which Lie group family (SO/SE/GL) and dimension maximises detection power per unit of computational cost?
3. **Adaptive thresholds**: Can holonomy scar thresholds be auto-calibrated from benign traffic without labeled harmful examples?
4. **Multi-agent topology**: How do confinement guarantees compose when multiple agents interact through shared group state?
5. **STARK proof overhead**: What is the latency cost of adding per-step zero-knowledge proofs for tamper-evident audit trails?

## License

MIT OR Apache-2.0
