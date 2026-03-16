# Holonomy-Centric GravRail Enhancements Design

**Date:** 2026-03-15
**Status:** Approved
**Goal:** Wire holonomy detection into the proxy (input + output), add threshold calibration, validate via jailbreak benchmark, and persist trajectory data for audit.

---

## Problem

Three gaps exist after the GloVe-50 embedding work:

1. **Holonomy is unobservable** — `Connection::holonomy()` exists but nothing calls it during live proxy operation. Manipulation that unfolds over a sequence of prompts is invisible.
2. **Thresholds are guesses** — `max_state_norm` is hardcoded. There is no data-driven way to set it, and there is no holonomy threshold at all.
3. **Sessions are ephemeral** — proxy state vanishes on restart. No audit trail of what happened in a session.

---

## Architecture

The four features form a dependency chain:

```text
holonomy in proxy → calibrate sets thresholds → benchmark validates → persistence records
```

Build order matches this chain: 1 → 2 → 3 → 4.

---

## Feature 1: Holonomy in Proxy (Input + Output)

The proxy processes **both sides** of every exchange:

- **Input side**: user message → `map_to_algebra` → input holonomy window → optional pre-LLM block
- **Output side**: LLM response → `map_to_algebra` → output holonomy window → optional post-LLM block (existing path)
- **Drift signal**: cosine distance between input and output algebra vectors (large gap = LLM drifted from topic)

### What changes

`ConfinementPipeline` (`src/proxy/confine.rs`) gains:

```rust
// Output-side (existing, extended)
output_holonomy_window: VecDeque<Vec<f64>>,
holonomy_window_size: usize,          // default 8, shared

// Input-side (new)
input_holonomy_window: VecDeque<Vec<f64>>,
last_input_coeffs: Option<Vec<f64>>,  // retained for drift computation
```

**Request path (new):**

1. Parse incoming request body; extract last user message text
2. Run `map_to_algebra(user_text, algebra_dim, scale)` → `input_coeffs`
3. Push `input_coeffs` into `input_holonomy_window`; store as `last_input_coeffs`
4. Compute input holonomy (scalar) and input norm
5. If `--input-holonomy-threshold` set and input holonomy exceeds it → return `429` with `X-GravRail-Block: input-holonomy` **before forwarding to upstream**
6. If `--input-norm-threshold` set and input norm exceeds it → return `429` with `X-GravRail-Block: input-norm`
7. Otherwise forward to upstream

**Response path (extended):**

1. Run existing `confine()` on LLM response → `output_coeffs`
2. Push into `output_holonomy_window`, compute output holonomy
3. Compute drift: `cosine_distance(last_input_coeffs, output_coeffs)`
4. If `--holonomy-threshold` set and output holonomy exceeds it → `429` with `X-GravRail-Block: output-holonomy`

### Response headers

```text
X-GravRail-Input-Norm: 4.2
X-GravRail-Input-Holonomy: 0.08
X-GravRail-Holonomy: 0.12           ← output-side
X-GravRail-Drift: 0.31              ← cosine distance input↔output
X-GravRail-Block: input-holonomy    ← only present when blocked
```

### New proxy CLI flags

```text
--input-holonomy-threshold <f64>    (default: 0.0 = report-only)
--input-norm-threshold <f64>        (default: 0.0 = report-only)
--holonomy-threshold <f64>          (default: 0.0 = report-only, output-side)
```

### What does NOT change

- `confine()` signature unchanged
- Existing `X-GravRail-Seq`, `X-GravRail-State`, `X-GravRail-Proof` headers unchanged

---

## Feature 2: Config File + `gravrail calibrate`

### Config file

`~/.gravrail/config.toml`:

```toml
[proxy]
max_state_norm = 12.4
holonomy_threshold = 0.85
input_holonomy_threshold = 0.60
input_norm_threshold = 10.0
holonomy_window = 8
```

Proxy reads this on startup. CLI flags override config values. Missing keys fall back to hardcoded defaults.

### `gravrail calibrate` command

1. Starts an embedded mock upstream (axum handler returning `{"choices":[{"message":{"content":"ok"}}]}`)
2. Spawns `gravrail proxy --json-startup --port 0 --upstream-url http://127.0.0.1:<mock_port>`
3. Reads `{"port":N,"token":"..."}` from proxy stdout
4. Sends 50 benign prompts (hardcoded corpus) through `/v1/chat/completions`
5. Collects all four `X-GravRail-*` metrics from each response
6. Computes `mean + 3σ` for each metric
7. Writes results to `~/.gravrail/config.toml`
8. Kills proxy and mock, prints summary

The calibration corpus is a hardcoded `Vec<&str>` of 50 neutral prompts (factual questions, coding help, creative writing requests).

---

## Feature 3: `gravrail benchmark`

Same subprocess pattern as calibrate. Tests 40 prompts:

- **20 benign**: factual, creative, coding — drawn from the calibrate corpus
- **20 jailbreak**: prompt injection, role-play bypasses, "ignore previous instructions", phishing, credential extraction — drawn from public AdvBench-style datasets

The embedded mock upstream echoes the prompt text back as the assistant response, so the confinement pipeline processes the actual jailbreak/benign text through both input and output paths.

### Output

```text
GravRail Benchmark
────────────────────────────────────────────────────────
                    Benign    Jailbreak   Separation
Input Norm (mean)     4.2        11.8        2.8×
Input Holonomy        0.12       0.74        6.2×
Output Norm           3.9        11.2        2.9×
Output Holonomy       0.09       0.68        7.6×
Drift (in↔out)        0.08       0.44        5.5×
────────────────────────────────────────────────────────
Suggested input_holonomy_threshold:  0.45
Suggested input_norm_threshold:      8.0
Suggested holonomy_threshold:        0.40

Run `gravrail calibrate` to apply these thresholds.
```

### Caveats (printed by the tool)

- Tests prompt text echoed as LLM response — production accuracy depends on actual LLM output patterns
- Holonomy is most sensitive to multi-turn manipulation; single-turn values may understate real-world detection rates
- Corpus is public; novel adversarial prompts using neutral language may not be detected by norm/holonomy alone

---

## Feature 4: Trajectory Persistence

### Schema

New `trajectory_steps` table in the existing SQLite database:

```sql
CREATE TABLE trajectory_steps (
    id           INTEGER PRIMARY KEY,
    circuit_id   TEXT    NOT NULL,
    session_id   TEXT    NOT NULL,  -- UUID generated at proxy startup
    step         INTEGER NOT NULL,
    input_norm   REAL,
    input_hol    REAL,
    output_norm  REAL    NOT NULL,
    output_hol   REAL,              -- NULL if window not yet full
    drift        REAL,
    algebra      BLOB    NOT NULL,  -- output Vec<f64> as f64 LE bytes
    ts           INTEGER NOT NULL   -- Unix ms
);
```

### Write path

After each exchange (input processed + response returned), `tokio::spawn` a fire-and-forget task that inserts one row. Never blocks the response path.

### `gravrail replay --session <id>`

Reads rows for the session and prints:

```text
Session abc123 · circuit my-circuit · 47 steps
Step  In-Norm  In-Hol  Out-Norm  Out-Hol  Drift   Timestamp
   1    3.2     0.05     3.1      —        0.04   2026-03-15 14:01:00
   2    4.1     0.08     3.9      0.07     0.11   2026-03-15 14:01:03
  ...
  31   14.7     0.91⚠   14.2     0.88⚠    0.62   2026-03-15 14:08:22
```

A `⚠` marker appears on any step where a metric exceeded the configured threshold.

---

## Summary

| Task | Deliverable |
|------|-------------|
| 1 | Input + output holonomy headers, drift signal, blocking flags |
| 2 | `~/.gravrail/config.toml` + `gravrail calibrate` command |
| 3 | `gravrail benchmark` with 40-prompt corpus, input+output metrics |
| 4 | `trajectory_steps` table + `gravrail replay` |

---

## Open Questions (post-implementation)

1. Does input holonomy separate benign from jailbreak better than output holonomy in practice?
2. Should window size be per-circuit (stored in config) or global?
3. Consider adding a `gravrail monitor` live-tail mode that streams `replay` output in real time.
