# GravRail

Deterministic geometric confinement for AI agents via Lie groups, gauge theory, and group-native cryptography.

## Architecture

Four domain layers, each building on the last:

| Layer | Module | Purpose |
|-------|--------|---------|
| **Lie** | `lie::group`, `lie::algebra`, `lie::represent`, `lie::invariants` | SO/SE/GL groups, exp/log maps, Killing form, Casimir operator |
| **Gauge** | `gauge::bundle`, `gauge::connection`, `gauge::curvature`, `gauge::invariance` | Fiber bundles, parallel transport, holonomy, gauge invariance checks |
| **Crypto** | `crypto::commit`, `crypto::sign`, `crypto::zkproof`, `crypto::keyexchange` | Pedersen commitments, Schnorr signatures, ZK proofs, DH key exchange |
| **Circuit** | `circuit::define`, `circuit::agent`, `circuit::map`, `circuit::reachability` | Confinement spaces, agent runners, text→algebra chokepoint, bounded BFS |

Infrastructure: `error`, `config`, `state` (SQLite), `lineage` (Merkle hash chain), `feedback` (adaptive suppression).

## Core Invariant

**Every LLM output goes through:** `text → map_to_algebra → constrain → exp → multiply`

The exponential map **always** produces an on-group element. Group multiplication **always** stays on-group (closure axiom). This is the confinement guarantee — it cannot be broken by any input.

## MCP Tools

All tools are prefixed `grav_*`. Key tools:

| Tool | What it does |
|------|-------------|
| `grav_circuit_create` | Define a confinement space (group + optional generator mask) |
| `grav_agent_create` | Spawn an agent in a circuit |
| `grav_step` | THE core op: text → algebra → exp → on-group state update |
| `grav_exp` / `grav_log` | Exponential/logarithmic maps |
| `grav_holonomy` | Detect curvature/manipulation via parallel transport |
| `grav_sign` / `grav_verify_sign` | Schnorr signatures on Lie groups |
| `grav_lineage` / `grav_lineage_verify` | Tamper-proof audit trail |

## Running

```bash
# Initialize
gravrail init --data-dir ~/.gravrail

# Start MCP server (stdio)
gravrail serve --data-dir ~/.gravrail
```

## Testing

```bash
cargo test
```

24 tests across 8 files: group axioms (proptest), algebra identities (proptest), representation homomorphism (proptest), gauge holonomy, crypto binding/soundness (proptest), circuit confinement (proptest), integration negotiation, manipulation detection, lineage tamper detection.

## Key Design Decisions

- **Exact Rodrigues formula** for SO(3) exp/log instead of Padé approximation — avoids 5e-8 numerical drift
- **Algebra-level verification** for Schnorr/ZK — avoids BCH non-commutativity issues in non-abelian groups
- **u64 normalization** for hash-to-f64 — `f64::from_le_bytes` produces extreme magnitudes where `% 1.0` degenerates to 0
- **SVD projection** after every group operation to keep elements on-manifold despite floating-point drift
