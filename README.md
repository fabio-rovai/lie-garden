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

## License

MIT OR Apache-2.0
