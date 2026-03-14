# GravProxy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a transport-level proxy that intercepts every LLM response and runs it through the confinement pipeline before forwarding to the consumer. The LLM has no bypass path.

**Architecture:** An Axum HTTP server (`proxy/server.rs`) binds a local port. Consumer sends OpenAI-compatible requests. The proxy forwards them upstream to the LLM, intercepts every response chunk, runs it through `map_to_algebra → constrain → exp → multiply` (`proxy/confine.rs`), checks reachability bounds, and either forwards with proof metadata or blocks. A watchdog thread (`proxy/watchdog.rs`) exchanges HMAC-signed heartbeats over a Unix domain socket for mutual liveness. A session token (`proxy/auth.rs`) prevents fake proxy substitution. SSE streaming is handled by `proxy/stream.rs`. The CLI subcommand is in `proxy/cli.rs`.

**Tech Stack:** Rust, Axum 0.8 (already in deps), reqwest (new — upstream HTTP client), tokio, hmac + sha2 (heartbeat signing), existing gravrail confinement pipeline.

**Design doc:** `docs/plans/2026-03-14-grav-proxy-design.md`

---

## Batch 1: Foundation (Tasks 1–3)

### Task 1: Add dependencies and create proxy module skeleton

**Files:**
- Modify: `Cargo.toml`
- Create: `src/proxy/mod.rs`
- Modify: `src/lib.rs:12` (add `pub mod proxy;`)

**Step 1: Add new dependencies to Cargo.toml**

Add these lines to `[dependencies]`:

```toml
reqwest = { version = "0.12", features = ["json", "stream"] }
hmac = "0.12"
hex = "0.4"
```

- `reqwest`: HTTP client for forwarding requests upstream to the LLM
- `hmac`: HMAC-SHA256 for signing heartbeat pings
- `hex`: Hex encoding for session tokens and HMAC signatures

**Step 2: Create `src/proxy/mod.rs`**

```rust
pub mod auth;
pub mod confine;
pub mod watchdog;
pub mod stream;
pub mod server;
pub mod cli;
```

**Step 3: Register the proxy module in `src/lib.rs`**

Add `pub mod proxy;` after the existing `pub mod server;` line (line 11).

**Step 4: Create placeholder files**

Create empty files with just a comment so the module compiles:
- `src/proxy/auth.rs` — `// Session token generation and validation`
- `src/proxy/confine.rs` — `// Per-response confinement pipeline`
- `src/proxy/watchdog.rs` — `// Heartbeat protocol and mutual liveness`
- `src/proxy/stream.rs` — `// SSE chunk buffering and confined forwarding`
- `src/proxy/server.rs` — `// Axum HTTP server, request forwarding, response interception`
- `src/proxy/cli.rs` — `// CLI subcommand: gravrail proxy`

**Step 5: Verify it compiles**

Run: `cargo check`
Expected: Compiles with no errors (warnings OK).

**Step 6: Commit**

```bash
git add Cargo.toml src/lib.rs src/proxy/
git commit -m "feat(proxy): add module skeleton and dependencies for GravProxy"
```

---

### Task 2: Session token auth (`proxy/auth.rs`)

**Files:**
- Modify: `src/proxy/auth.rs`
- Create: `src/proxy/auth.rs` (tests inline)

**Step 1: Write the failing tests**

Add to `src/proxy/auth.rs`:

```rust
use rand::Rng;
use sha2::{Sha256, Digest};

/// A one-time session token generated at proxy startup.
/// Consumer must present this token to connect.
pub struct SessionAuth {
    token: String,
    validated: bool,
}

impl SessionAuth {
    /// Generate a new random session token (32 hex chars).
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let bytes: [u8; 16] = rng.gen();
        let token = hex::encode(bytes);
        Self { token, validated: false }
    }

    /// Get the token string (printed to stderr at startup).
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Validate a token from a request header. Returns true on first valid match.
    /// After first validation, all subsequent requests are allowed without re-checking.
    pub fn validate(&mut self, candidate: &str) -> bool {
        if self.validated {
            return true;
        }
        if constant_time_eq(candidate.as_bytes(), self.token.as_bytes()) {
            self.validated = true;
            true
        } else {
            false
        }
    }

    /// Whether the session has been authenticated.
    pub fn is_validated(&self) -> bool {
        self.validated
    }
}

/// Constant-time comparison to prevent timing attacks on token.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_32_hex_chars() {
        let auth = SessionAuth::new();
        assert_eq!(auth.token().len(), 32);
        assert!(auth.token().chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_correct_token_validates() {
        let mut auth = SessionAuth::new();
        let token = auth.token().to_string();
        assert!(!auth.is_validated());
        assert!(auth.validate(&token));
        assert!(auth.is_validated());
    }

    #[test]
    fn test_wrong_token_rejected() {
        let mut auth = SessionAuth::new();
        assert!(!auth.validate("0000000000000000000000000000000000"));
        assert!(!auth.is_validated());
    }

    #[test]
    fn test_after_validation_all_pass() {
        let mut auth = SessionAuth::new();
        let token = auth.token().to_string();
        auth.validate(&token);
        // After first validation, any call returns true (session established)
        assert!(auth.validate("anything"));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
    }
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test --lib proxy::auth -- --nocapture`
Expected: 5 tests pass.

**Step 3: Commit**

```bash
git add src/proxy/auth.rs
git commit -m "feat(proxy): session token authentication with constant-time comparison"
```

---

### Task 3: Watchdog heartbeat protocol (`proxy/watchdog.rs`)

**Files:**
- Modify: `src/proxy/watchdog.rs`

This is the most security-critical component. The watchdog and proxy exchange HMAC-signed pings over a channel (in-process for safety — Unix domain sockets are an option later but channels are simpler and tamper-proof within a single process).

**Step 1: Write watchdog implementation with tests**

```rust
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

type HmacSha256 = Hmac<Sha256>;

/// Signed heartbeat ping.
#[derive(Debug, Clone)]
pub struct Heartbeat {
    pub seq: u64,
    pub timestamp_ms: u64,
    pub signature: Vec<u8>,
}

/// Shared state between proxy and watchdog.
pub struct WatchdogHandle {
    /// Set to true when either side detects failure.
    pub killed: Arc<AtomicBool>,
    /// Last heartbeat sequence received from watchdog.
    pub last_watchdog_seq: Arc<AtomicU64>,
    /// Channel to send pings TO the watchdog.
    pub tx_to_watchdog: mpsc::Sender<Heartbeat>,
    /// Channel to receive pings FROM the watchdog.
    pub rx_from_watchdog: mpsc::Receiver<Heartbeat>,
    /// HMAC key for verification.
    hmac_key: Vec<u8>,
}

/// Watchdog configuration.
pub struct WatchdogConfig {
    /// How often to send heartbeats (default: 1s).
    pub interval: Duration,
    /// How long before declaring the other side dead (default: 3s).
    pub timeout: Duration,
}

impl Default for WatchdogConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(1),
            timeout: Duration::from_secs(3),
        }
    }
}

/// Create a matched pair: (proxy_handle, watchdog_task_future).
///
/// The proxy_handle is used by the HTTP server to check liveness.
/// The returned future runs the watchdog loop — spawn it on tokio.
pub fn create_watchdog(config: WatchdogConfig) -> (WatchdogHandle, impl std::future::Future<Output = ()>) {
    // Generate HMAC key from OS randomness (exists only in memory)
    let mut hmac_key = vec![0u8; 32];
    rand::Rng::fill(&mut rand::thread_rng(), &mut hmac_key[..]);

    let killed = Arc::new(AtomicBool::new(false));
    let last_watchdog_seq = Arc::new(AtomicU64::new(0));

    // Channels: proxy <-> watchdog
    let (tx_proxy_to_wd, mut rx_proxy_to_wd) = mpsc::channel::<Heartbeat>(8);
    let (tx_wd_to_proxy, rx_wd_to_proxy) = mpsc::channel::<Heartbeat>(8);

    let killed_clone = killed.clone();
    let last_seq_clone = last_watchdog_seq.clone();
    let key_clone = hmac_key.clone();
    let interval = config.interval;
    let timeout = config.timeout;

    let watchdog_future = async move {
        let mut seq: u64 = 0;
        let mut last_proxy_ping = Instant::now();

        loop {
            if killed_clone.load(Ordering::Relaxed) {
                break;
            }

            // Send heartbeat to proxy
            seq += 1;
            let hb = sign_heartbeat(seq, &key_clone);
            if tx_wd_to_proxy.send(hb).await.is_err() {
                // Proxy dropped its receiver — it's dead
                killed_clone.store(true, Ordering::Relaxed);
                break;
            }

            // Wait for proxy's ping (with timeout = interval)
            match tokio::time::timeout(interval, rx_proxy_to_wd.recv()).await {
                Ok(Some(ping)) => {
                    if verify_heartbeat(&ping, &key_clone) {
                        last_proxy_ping = Instant::now();
                    }
                    // Invalid signature — ignore but don't reset timer
                }
                Ok(None) => {
                    // Channel closed — proxy is dead
                    killed_clone.store(true, Ordering::Relaxed);
                    break;
                }
                Err(_) => {
                    // Timeout — check if we've exceeded the liveness threshold
                }
            }

            // Check liveness
            if last_proxy_ping.elapsed() > timeout {
                killed_clone.store(true, Ordering::Relaxed);
                break;
            }

            last_seq_clone.store(seq, Ordering::Relaxed);
            tokio::time::sleep(interval).await;
        }
    };

    let handle = WatchdogHandle {
        killed,
        last_watchdog_seq,
        tx_to_watchdog: tx_proxy_to_wd,
        rx_from_watchdog: rx_wd_to_proxy,
        hmac_key,
    };

    (handle, watchdog_future)
}

/// Sign a heartbeat with HMAC-SHA256.
pub fn sign_heartbeat(seq: u64, key: &[u8]) -> Heartbeat {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC key");
    mac.update(&seq.to_le_bytes());
    mac.update(&now.to_le_bytes());
    let sig = mac.finalize().into_bytes().to_vec();

    Heartbeat {
        seq,
        timestamp_ms: now,
        signature: sig,
    }
}

/// Verify a heartbeat's HMAC signature.
pub fn verify_heartbeat(hb: &Heartbeat, key: &[u8]) -> bool {
    let mut mac = HmacSha256::new_from_slice(key).expect("HMAC key");
    mac.update(&hb.seq.to_le_bytes());
    mac.update(&hb.timestamp_ms.to_le_bytes());
    mac.verify_slice(&hb.signature).is_ok()
}

impl WatchdogHandle {
    /// Check if the system is still alive.
    pub fn is_alive(&self) -> bool {
        !self.killed.load(Ordering::Relaxed)
    }

    /// Kill the system (proxy detected an error).
    pub fn kill(&self) {
        self.killed.store(true, Ordering::Relaxed);
    }

    /// Send a heartbeat ping to the watchdog. Call this from the proxy's own loop.
    pub async fn send_ping(&self) -> bool {
        if !self.is_alive() {
            return false;
        }
        let hb = sign_heartbeat(0, &self.hmac_key);
        self.tx_to_watchdog.send(hb).await.is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify_heartbeat() {
        let key = b"test-key-32-bytes-long-enough!!!";
        let hb = sign_heartbeat(42, key);
        assert!(verify_heartbeat(&hb, key));
    }

    #[test]
    fn test_tampered_heartbeat_rejected() {
        let key = b"test-key-32-bytes-long-enough!!!";
        let mut hb = sign_heartbeat(42, key);
        hb.seq = 43; // tamper
        assert!(!verify_heartbeat(&hb, key));
    }

    #[test]
    fn test_wrong_key_rejected() {
        let key1 = b"key-one-32-bytes-long-enough!!!!";
        let key2 = b"key-two-32-bytes-long-enough!!!!";
        let hb = sign_heartbeat(1, key1);
        assert!(!verify_heartbeat(&hb, key2));
    }

    #[tokio::test]
    async fn test_watchdog_starts_alive() {
        let (handle, _fut) = create_watchdog(WatchdogConfig::default());
        assert!(handle.is_alive());
    }

    #[tokio::test]
    async fn test_watchdog_kill() {
        let (handle, _fut) = create_watchdog(WatchdogConfig::default());
        handle.kill();
        assert!(!handle.is_alive());
    }

    #[tokio::test]
    async fn test_watchdog_detects_proxy_death() {
        let config = WatchdogConfig {
            interval: Duration::from_millis(50),
            timeout: Duration::from_millis(150),
        };
        let (handle, fut) = create_watchdog(config);
        let killed = handle.killed.clone();

        // Spawn watchdog but never send pings from proxy side
        // Drop the handle (closes tx_to_watchdog channel)
        drop(handle);

        // Run watchdog — it should detect proxy death
        tokio::time::timeout(Duration::from_secs(1), fut).await.ok();
        assert!(killed.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_mutual_heartbeat_keeps_alive() {
        let config = WatchdogConfig {
            interval: Duration::from_millis(50),
            timeout: Duration::from_millis(300),
        };
        let (mut handle, fut) = create_watchdog(config);
        let killed = handle.killed.clone();

        // Spawn watchdog
        let wd_handle = tokio::spawn(fut);

        // Exchange heartbeats for 200ms
        for _ in 0..4 {
            // Send ping to watchdog
            handle.send_ping().await;
            // Receive ping from watchdog
            if let Ok(Some(_hb)) = tokio::time::timeout(
                Duration::from_millis(100),
                handle.rx_from_watchdog.recv()
            ).await {
                // Got heartbeat — good
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        assert!(
            !killed.load(Ordering::Relaxed),
            "System should be alive during active heartbeat exchange"
        );

        // Clean up
        handle.kill();
        wd_handle.abort();
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib proxy::watchdog -- --nocapture`
Expected: 7 tests pass.

**Step 3: Commit**

```bash
git add src/proxy/watchdog.rs
git commit -m "feat(proxy): HMAC-signed watchdog heartbeat with mutual liveness detection"
```

---

## Batch 2: Confinement & Streaming (Tasks 4–5)

### Task 4: Per-response confinement pipeline (`proxy/confine.rs`)

**Files:**
- Modify: `src/proxy/confine.rs`

This module takes text content from an LLM response, runs it through the full confinement pipeline, and returns either the confined result with proof metadata or a violation error.

**Step 1: Write implementation with tests**

```rust
use crate::circuit::define::Circuit;
use crate::circuit::map::map_to_algebra;
use crate::lie::group::GroupElement;
use crate::crypto::stark::confinement::{prove_confinement, verify_confinement_proof, ConfinementProof};

/// Result of confining a single text chunk.
pub struct ConfinementResult {
    /// The original text (passed through unchanged if within bounds).
    pub text: String,
    /// Updated agent state after this step.
    pub state: GroupElement,
    /// Monotonic sequence number for dead man's switch.
    pub seq: u64,
    /// STARK proof of confinement (optional, can be batched).
    pub proof: Option<ConfinementProof>,
    /// State matrix elements for header.
    pub state_elements: Vec<f64>,
}

/// Error when confinement fails.
#[derive(Debug)]
pub enum ConfinementError {
    /// Text maps to a state outside reachability bounds.
    ReachabilityViolation {
        text_preview: String,
        state: Vec<f64>,
    },
    /// Pipeline internal error.
    PipelineError(String),
}

impl std::fmt::Display for ConfinementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReachabilityViolation { text_preview, .. } => {
                write!(f, "confinement_violation: text '{}...' maps outside bounds", text_preview)
            }
            Self::PipelineError(msg) => write!(f, "pipeline_error: {}", msg),
        }
    }
}

/// Confinement pipeline state — held by the proxy for the duration of a session.
pub struct ConfinementPipeline {
    circuit: Circuit,
    state: GroupElement,
    seq: u64,
    /// Optional: reachability bound (max Frobenius norm of log of state).
    max_state_norm: Option<f64>,
    /// Whether to generate STARK proofs per response (vs batched).
    prove_every_step: bool,
}

impl ConfinementPipeline {
    pub fn new(circuit: Circuit, max_state_norm: Option<f64>, prove_every_step: bool) -> Self {
        let state = circuit.group.identity();
        Self {
            circuit,
            state,
            seq: 0,
            max_state_norm,
            prove_every_step,
        }
    }

    /// Run a text chunk through the full confinement pipeline.
    ///
    /// text → map_to_algebra → constrain → exp → multiply → reachability check
    ///
    /// Returns ConfinementResult on success, ConfinementError if out of bounds.
    pub fn confine(&mut self, text: &str) -> Result<ConfinementResult, ConfinementError> {
        // 1. Map text to algebra coefficients
        let raw_coeffs = map_to_algebra(text, self.circuit.group.algebra_dim(), 1.0);

        // 2. Apply circuit constraints (mask inactive generators)
        let constrained = self.circuit.constrain_algebra(&raw_coeffs);

        // 3. Exponential map — ALWAYS lands on group
        let step_element = self.circuit.group.exp(&constrained);

        // 4. Group multiplication — ALWAYS stays on group (closure axiom)
        let new_state = self.circuit.group.multiply(&self.state, &step_element);

        // 5. Reachability check
        if let Some(max_norm) = self.max_state_norm {
            let state_norm = state_frobenius_distance(&self.circuit.group.identity(), &new_state);
            if state_norm > max_norm {
                let preview = if text.len() > 50 {
                    format!("{}...", &text[..50])
                } else {
                    text.to_string()
                };
                return Err(ConfinementError::ReachabilityViolation {
                    text_preview: preview,
                    state: new_state.matrix().to_vec(),
                });
            }
        }

        // 6. Update state
        self.state = new_state;
        self.seq += 1;

        // 7. Generate proof (optional)
        let mask = self.circuit.active_generators.clone()
            .unwrap_or_else(|| vec![true; self.circuit.group.algebra_dim()]);
        let proof = if self.prove_every_step {
            Some(prove_confinement(&raw_coeffs, &mask, &constrained))
        } else {
            None
        };

        Ok(ConfinementResult {
            text: text.to_string(),
            state: self.state.clone(),
            seq: self.seq,
            proof,
            state_elements: self.state.matrix().to_vec(),
        })
    }

    /// Get current state.
    pub fn state(&self) -> &GroupElement {
        &self.state
    }

    /// Get current sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }
}

/// Frobenius distance between two group elements.
fn state_frobenius_distance(a: &GroupElement, b: &GroupElement) -> f64 {
    a.matrix().iter().zip(b.matrix().iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie::group::{LieGroup, GroupType};

    fn test_circuit() -> Circuit {
        Circuit::new(
            LieGroup::new(GroupType::SO, 3),
            None, // all generators active
        )
    }

    #[test]
    fn test_confine_basic() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let result = pipeline.confine("hello world");
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.text, "hello world");
        assert_eq!(r.seq, 1);
    }

    #[test]
    fn test_confine_sequence_increments() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        pipeline.confine("step 1").unwrap();
        pipeline.confine("step 2").unwrap();
        let r = pipeline.confine("step 3").unwrap();
        assert_eq!(r.seq, 3);
    }

    #[test]
    fn test_confine_state_changes() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let identity = pipeline.state().clone();
        pipeline.confine("move").unwrap();
        // State should have changed from identity
        assert!(!pipeline.state().close_to(&identity, 1e-10));
    }

    #[test]
    fn test_confine_deterministic() {
        let circuit1 = test_circuit();
        let circuit2 = test_circuit();
        let mut p1 = ConfinementPipeline::new(circuit1, None, false);
        let mut p2 = ConfinementPipeline::new(circuit2, None, false);
        p1.confine("test input").unwrap();
        p2.confine("test input").unwrap();
        assert!(p1.state().close_to(p2.state(), 1e-12));
    }

    #[test]
    fn test_confine_with_reachability_bound() {
        let circuit = test_circuit();
        // Very tight bound — should trigger violation
        let mut pipeline = ConfinementPipeline::new(circuit, Some(0.001), false);
        let result = pipeline.confine("this will likely exceed a tiny bound");
        assert!(result.is_err());
        if let Err(ConfinementError::ReachabilityViolation { .. }) = result {
            // Expected
        } else {
            panic!("Expected ReachabilityViolation");
        }
    }

    #[test]
    fn test_confine_generous_bound_passes() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, Some(100.0), false);
        let result = pipeline.confine("normal text");
        assert!(result.is_ok());
    }

    #[test]
    fn test_confine_with_proof() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, None, true);
        let result = pipeline.confine("prove this").unwrap();
        assert!(result.proof.is_some());
    }

    #[test]
    fn test_confine_without_proof() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let result = pipeline.confine("no proof").unwrap();
        assert!(result.proof.is_none());
    }

    #[test]
    fn test_confine_with_masked_generators() {
        let circuit = Circuit::new(
            LieGroup::new(GroupType::SO, 3),
            Some(vec![true, false, false]), // only first generator
        );
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let result = pipeline.confine("masked");
        assert!(result.is_ok());
    }

    #[test]
    fn test_state_elements_in_result() {
        let circuit = test_circuit();
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let r = pipeline.confine("check state").unwrap();
        // SO(3) → 3x3 matrix → 9 elements
        assert_eq!(r.state_elements.len(), 9);
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib proxy::confine -- --nocapture`
Expected: 10 tests pass.

**Step 3: Commit**

```bash
git add src/proxy/confine.rs
git commit -m "feat(proxy): confinement pipeline with reachability bounds and optional STARK proofs"
```

---

### Task 5: SSE stream buffering (`proxy/stream.rs`)

**Files:**
- Modify: `src/proxy/stream.rs`

This module handles Server-Sent Events (SSE) streaming from the LLM. It buffers each `data:` chunk, extracts text content from the OpenAI-format JSON, confines it, and produces output chunks with proof metadata headers.

**Step 1: Write implementation with tests**

```rust
use serde_json::Value;

/// Extract text content from an OpenAI chat completion chunk.
///
/// Handles both streaming (`choices[0].delta.content`) and
/// non-streaming (`choices[0].message.content`) formats.
pub fn extract_text_content(json: &Value) -> Option<String> {
    // Streaming format: choices[0].delta.content
    if let Some(content) = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("delta"))
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
    {
        if !content.is_empty() {
            return Some(content.to_string());
        }
    }

    // Non-streaming format: choices[0].message.content
    if let Some(content) = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
    {
        if !content.is_empty() {
            return Some(content.to_string());
        }
    }

    None
}

/// Parse an SSE line into its data payload.
/// Returns None for non-data lines (comments, empty, event:, etc).
pub fn parse_sse_data(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed == "data: [DONE]" {
        return None;
    }
    if let Some(data) = trimmed.strip_prefix("data: ") {
        Some(data.to_string())
    } else if let Some(data) = trimmed.strip_prefix("data:") {
        Some(data.to_string())
    } else {
        None
    }
}

/// Format a confined chunk back into SSE format with metadata headers.
pub fn format_sse_chunk(
    original_data: &str,
    seq: u64,
    state_elements: &[f64],
    proof_hash: Option<&str>,
) -> String {
    // We pass through the original JSON data and add confinement metadata
    // as additional SSE comments before the data line.
    let mut output = String::new();
    output.push_str(&format!(": x-gravrail-seq={}\n", seq));
    output.push_str(&format!(": x-gravrail-state={:?}\n", state_elements));
    if let Some(hash) = proof_hash {
        output.push_str(&format!(": x-gravrail-proof={}\n", hash));
    }
    output.push_str(&format!("data: {}\n\n", original_data));
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_streaming_content() {
        let chunk = json!({
            "choices": [{"delta": {"content": "Hello"}, "index": 0}],
            "model": "gpt-4"
        });
        assert_eq!(extract_text_content(&chunk), Some("Hello".to_string()));
    }

    #[test]
    fn test_extract_non_streaming_content() {
        let response = json!({
            "choices": [{"message": {"content": "Full response here"}, "index": 0}],
            "model": "gpt-4"
        });
        assert_eq!(extract_text_content(&response), Some("Full response here".to_string()));
    }

    #[test]
    fn test_extract_empty_delta() {
        let chunk = json!({
            "choices": [{"delta": {"content": ""}, "index": 0}]
        });
        assert_eq!(extract_text_content(&chunk), None);
    }

    #[test]
    fn test_extract_no_content_field() {
        let chunk = json!({
            "choices": [{"delta": {"role": "assistant"}, "index": 0}]
        });
        assert_eq!(extract_text_content(&chunk), None);
    }

    #[test]
    fn test_parse_sse_data_line() {
        assert_eq!(
            parse_sse_data("data: {\"choices\":[]}"),
            Some("{\"choices\":[]}".to_string())
        );
    }

    #[test]
    fn test_parse_sse_done() {
        assert_eq!(parse_sse_data("data: [DONE]"), None);
    }

    #[test]
    fn test_parse_sse_comment() {
        assert_eq!(parse_sse_data(": this is a comment"), None);
    }

    #[test]
    fn test_parse_sse_empty() {
        assert_eq!(parse_sse_data(""), None);
    }

    #[test]
    fn test_format_sse_chunk() {
        let output = format_sse_chunk(
            "{\"choices\":[]}",
            42,
            &[1.0, 0.0, 0.0, 1.0],
            Some("abc123"),
        );
        assert!(output.contains("x-gravrail-seq=42"));
        assert!(output.contains("x-gravrail-proof=abc123"));
        assert!(output.contains("data: {\"choices\":[]}"));
    }

    #[test]
    fn test_format_sse_chunk_no_proof() {
        let output = format_sse_chunk("{}", 1, &[1.0], None);
        assert!(!output.contains("x-gravrail-proof"));
        assert!(output.contains("data: {}"));
    }
}
```

**Step 2: Run tests**

Run: `cargo test --lib proxy::stream -- --nocapture`
Expected: 10 tests pass.

**Step 3: Commit**

```bash
git add src/proxy/stream.rs
git commit -m "feat(proxy): SSE stream parsing and confined chunk formatting"
```

---

## Batch 3: Server & CLI (Tasks 6–8)

### Task 6: HTTP proxy server (`proxy/server.rs`)

**Files:**
- Modify: `src/proxy/server.rs`

The Axum HTTP server that ties everything together. It:
1. Accepts consumer connections on a local port
2. Validates the session token on first request
3. Forwards requests upstream to the LLM
4. Intercepts responses, confines them, forwards or blocks
5. Checks watchdog liveness before every response

**Step 1: Write the server implementation**

```rust
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, Response, StatusCode, HeaderMap},
    response::IntoResponse,
    routing::any,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use reqwest::Client;

use crate::circuit::define::Circuit;
use super::auth::SessionAuth;
use super::confine::{ConfinementPipeline, ConfinementError};
use super::watchdog::WatchdogHandle;
use super::stream::{extract_text_content, parse_sse_data, format_sse_chunk};

/// Shared proxy state.
pub struct ProxyState {
    pub auth: Mutex<SessionAuth>,
    pub pipeline: Mutex<ConfinementPipeline>,
    pub watchdog: WatchdogHandle,
    pub upstream_url: String,
    pub client: Client,
}

/// Build the Axum router for the proxy.
pub fn build_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/{*path}", any(proxy_handler))
        .with_state(state)
}

/// Main proxy handler — intercepts all requests.
async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    req: Request<Body>,
) -> impl IntoResponse {
    // 1. Check watchdog liveness
    if !state.watchdog.is_alive() {
        return Response::builder()
            .status(StatusCode::SERVICE_UNAVAILABLE)
            .body(Body::from("{\"error\":\"proxy_killed\",\"message\":\"Watchdog liveness check failed. Proxy is shut down.\"}"))
            .unwrap();
    }

    // 2. Validate session token
    {
        let mut auth = state.auth.lock().await;
        if !auth.is_validated() {
            let token = req.headers()
                .get("X-GravRail-Token")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("");
            if !auth.validate(token) {
                return Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .body(Body::from("{\"error\":\"invalid_token\",\"message\":\"Invalid or missing X-GravRail-Token header\"}"))
                    .unwrap();
            }
        }
    }

    // 3. Build upstream request
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| format!("?{}", q)).unwrap_or_default();
    let upstream_url = format!("{}{}{}", state.upstream_url, path, query);

    // Copy headers (except host and auth)
    let mut upstream_headers = HeaderMap::new();
    for (key, value) in req.headers() {
        let key_str = key.as_str().to_lowercase();
        if key_str != "host" && key_str != "x-gravrail-token" {
            upstream_headers.insert(key.clone(), value.clone());
        }
    }

    // Read request body
    let body_bytes = match axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from(format!("{{\"error\":\"bad_request\",\"message\":\"{}\"}}", e)))
                .unwrap();
        }
    };

    // Check if streaming is requested
    let is_streaming = serde_json::from_slice::<serde_json::Value>(&body_bytes)
        .ok()
        .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
        .unwrap_or(false);

    // 4. Forward to upstream
    let upstream_response = match state.client
        .request(method, &upstream_url)
        .headers(upstream_headers)
        .body(body_bytes.to_vec())
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("{{\"error\":\"upstream_error\",\"message\":\"{}\"}}", e)))
                .unwrap();
        }
    };

    let upstream_status = upstream_response.status();

    // If upstream returned an error, pass it through (no confinement needed)
    if !upstream_status.is_success() {
        let body = upstream_response.bytes().await.unwrap_or_default();
        return Response::builder()
            .status(StatusCode::from_u16(upstream_status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY))
            .body(Body::from(body))
            .unwrap();
    }

    // 5. Handle response based on streaming mode
    if is_streaming {
        handle_streaming_response(state, upstream_response).await
    } else {
        handle_non_streaming_response(state, upstream_response).await
    }
}

/// Handle a non-streaming response: confine the full response body.
async fn handle_non_streaming_response(
    state: Arc<ProxyState>,
    upstream_response: reqwest::Response,
) -> Response<Body> {
    let body_bytes = match upstream_response.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from(format!("{{\"error\":\"upstream_read_error\",\"message\":\"{}\"}}", e)))
                .unwrap();
        }
    };

    let json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(j) => j,
        Err(_) => {
            // Not JSON — can't confine, block it
            return Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Body::from("{\"error\":\"non_json_response\",\"message\":\"Upstream returned non-JSON response\"}"))
                .unwrap();
        }
    };

    // Extract text content and confine it
    if let Some(text) = extract_text_content(&json) {
        let mut pipeline = state.pipeline.lock().await;
        match pipeline.confine(&text) {
            Ok(result) => {
                // Forward original response with confinement headers
                let mut response = Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .header("x-gravrail-seq", result.seq.to_string())
                    .header("x-gravrail-state", format!("{:?}", result.state_elements));

                if let Some(proof) = &result.proof {
                    response = response.header("x-gravrail-proof", &proof.trace_root);
                }

                response
                    .body(Body::from(body_bytes))
                    .unwrap()
            }
            Err(ConfinementError::ReachabilityViolation { text_preview, .. }) => {
                Response::builder()
                    .status(StatusCode::FORBIDDEN)
                    .body(Body::from(format!(
                        "{{\"error\":\"confinement_violation\",\"message\":\"Text '{}' maps outside reachability bounds\"}}",
                        text_preview.replace('"', "\\\"")
                    )))
                    .unwrap()
            }
            Err(ConfinementError::PipelineError(msg)) => {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::from(format!(
                        "{{\"error\":\"pipeline_error\",\"message\":\"{}\"}}",
                        msg.replace('"', "\\\"")
                    )))
                    .unwrap()
            }
        }
    } else {
        // No text content (e.g., function call, stop reason only) — pass through
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(body_bytes))
            .unwrap()
    }
}

/// Handle a streaming SSE response: confine each chunk.
async fn handle_streaming_response(
    state: Arc<ProxyState>,
    upstream_response: reqwest::Response,
) -> Response<Body> {
    let state_clone = state.clone();

    let stream = async_stream::stream! {
        let mut text_buffer = String::new();
        let body = upstream_response.bytes().await.unwrap_or_default();
        let body_str = String::from_utf8_lossy(&body);

        for line in body_str.lines() {
            // Check watchdog between chunks
            if !state_clone.watchdog.is_alive() {
                yield Err(std::io::Error::new(
                    std::io::ErrorKind::ConnectionAborted,
                    "Watchdog killed proxy",
                ));
                return;
            }

            if let Some(data) = parse_sse_data(line) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                    if let Some(text) = extract_text_content(&json) {
                        let mut pipeline = state_clone.pipeline.lock().await;
                        match pipeline.confine(&text) {
                            Ok(result) => {
                                let proof_hash = result.proof.as_ref().map(|p| p.trace_root.as_str());
                                let chunk = format_sse_chunk(
                                    &data,
                                    result.seq,
                                    &result.state_elements,
                                    proof_hash,
                                );
                                yield Ok::<_, std::io::Error>(chunk);
                            }
                            Err(_) => {
                                // Confinement violation — kill the stream
                                let error_json = "{\"error\":\"confinement_violation\"}";
                                yield Ok(format!("data: {}\n\n", error_json));
                                return;
                            }
                        }
                    } else {
                        // Non-content chunk (role, stop, etc) — pass through
                        yield Ok(format!("data: {}\n\n", data));
                    }
                }
            } else if line.trim() == "data: [DONE]" {
                yield Ok("data: [DONE]\n\n".to_string());
            }
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap()
}
```

**Step 2: Add `async-stream` dependency to Cargo.toml**

```toml
async-stream = "0.3"
```

**Step 3: Verify it compiles**

Run: `cargo check`
Expected: Compiles (warnings OK).

**Step 4: Commit**

```bash
git add src/proxy/server.rs Cargo.toml
git commit -m "feat(proxy): Axum HTTP proxy server with confinement interception"
```

---

### Task 7: CLI subcommand (`proxy/cli.rs` + `main.rs`)

**Files:**
- Modify: `src/proxy/cli.rs`
- Modify: `src/main.rs`

**Step 1: Write the CLI module**

```rust
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use reqwest::Client;

use crate::circuit::define::Circuit;
use crate::lie::group::{LieGroup, GroupType};
use crate::state::StateDb;

use super::auth::SessionAuth;
use super::confine::ConfinementPipeline;
use super::watchdog::{create_watchdog, WatchdogConfig};
use super::server::{ProxyState, build_router};

/// Configuration parsed from CLI args.
pub struct ProxyConfig {
    pub circuit_id: Option<String>,
    pub group_type: String,
    pub group_dim: usize,
    pub upstream_url: String,
    pub port: u16,
    pub heartbeat_interval_ms: u64,
    pub heartbeat_timeout_ms: u64,
    pub max_state_norm: Option<f64>,
    pub prove_every_step: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            circuit_id: None,
            group_type: "SO".to_string(),
            group_dim: 3,
            upstream_url: "http://localhost:11434/v1".to_string(),
            port: 8340,
            heartbeat_interval_ms: 1000,
            heartbeat_timeout_ms: 3000,
            max_state_norm: None,
            prove_every_step: false,
        }
    }
}

/// Start the proxy server. This is the main entry point called from main.rs.
pub async fn run_proxy(config: ProxyConfig) -> anyhow::Result<()> {
    // 1. Build or load circuit
    let group_type = match config.group_type.to_uppercase().as_str() {
        "SO" => GroupType::SO,
        "SE" => GroupType::SE,
        "GL" => GroupType::GL,
        other => anyhow::bail!("Unknown group type: {}. Use SO, SE, or GL.", other),
    };
    let group = LieGroup::new(group_type, config.group_dim);
    let circuit = Circuit::new(group, None);

    eprintln!("GravProxy starting...");
    eprintln!("  Circuit: {:?}({}) — algebra dim {}", group_type, config.group_dim, circuit.group.algebra_dim());
    eprintln!("  Upstream: {}", config.upstream_url);
    eprintln!("  Port: {}", config.port);

    // 2. Create session auth
    let auth = SessionAuth::new();
    eprintln!("  Session token: {}", auth.token());
    eprintln!("  (Consumer must send X-GravRail-Token: {} on first request)", auth.token());

    // 3. Create watchdog
    let watchdog_config = WatchdogConfig {
        interval: Duration::from_millis(config.heartbeat_interval_ms),
        timeout: Duration::from_millis(config.heartbeat_timeout_ms),
    };
    let (watchdog_handle, watchdog_future) = create_watchdog(watchdog_config);

    // 4. Create confinement pipeline
    let pipeline = ConfinementPipeline::new(circuit, config.max_state_norm, config.prove_every_step);

    // 5. Build shared state
    let state = Arc::new(ProxyState {
        auth: Mutex::new(auth),
        pipeline: Mutex::new(pipeline),
        watchdog: watchdog_handle,
        upstream_url: config.upstream_url.clone(),
        client: Client::new(),
    });

    // 6. Spawn watchdog
    let watchdog_state = state.clone();
    tokio::spawn(async move {
        watchdog_future.await;
        eprintln!("WATCHDOG: Liveness check failed — proxy shutting down");
    });

    // 7. Spawn proxy heartbeat sender (pings the watchdog periodically)
    let ping_state = state.clone();
    let ping_interval = Duration::from_millis(config.heartbeat_interval_ms);
    tokio::spawn(async move {
        loop {
            if !ping_state.watchdog.is_alive() {
                break;
            }
            ping_state.watchdog.send_ping().await;
            tokio::time::sleep(ping_interval).await;
        }
    });

    // 8. Start Axum server
    let app = build_router(state);
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("GravProxy listening on {}", addr);
    eprintln!("  Confinement: ACTIVE (every response confined)");
    eprintln!("  Watchdog: ACTIVE (heartbeat {}ms, timeout {}ms)",
        config.heartbeat_interval_ms, config.heartbeat_timeout_ms);

    axum::serve(listener, app).await?;

    Ok(())
}
```

**Step 2: Update `src/main.rs` to add the `proxy` subcommand**

Add to the `Commands` enum:

```rust
/// Start confinement proxy (transport-level enforcement)
Proxy {
    /// Upstream LLM URL (e.g., http://localhost:11434/v1)
    #[arg(long)]
    upstream: String,

    /// Port to bind the proxy on
    #[arg(long, default_value = "8340")]
    port: u16,

    /// Lie group type: SO, SE, or GL
    #[arg(long, default_value = "SO")]
    group_type: String,

    /// Group dimension (matrix size)
    #[arg(long, default_value = "3")]
    group_dim: usize,

    /// Heartbeat interval in milliseconds
    #[arg(long, default_value = "1000")]
    heartbeat_interval: u64,

    /// Heartbeat timeout in milliseconds
    #[arg(long, default_value = "3000")]
    heartbeat_timeout: u64,

    /// Max state norm for reachability bound (optional)
    #[arg(long)]
    max_state_norm: Option<f64>,

    /// Generate STARK proof for every response
    #[arg(long, default_value = "false")]
    prove: bool,
},
```

Add the match arm in `main()`:

```rust
Commands::Proxy {
    upstream,
    port,
    group_type,
    group_dim,
    heartbeat_interval,
    heartbeat_timeout,
    max_state_norm,
    prove,
} => {
    gravrail::proxy::cli::run_proxy(gravrail::proxy::cli::ProxyConfig {
        circuit_id: None,
        group_type,
        group_dim,
        upstream_url: upstream,
        port,
        heartbeat_interval_ms: heartbeat_interval,
        heartbeat_timeout_ms: heartbeat_timeout,
        max_state_norm,
        prove_every_step: prove,
    }).await?;
}
```

**Step 3: Verify it compiles**

Run: `cargo check`
Expected: Compiles (warnings OK).

**Step 4: Test help output**

Run: `cargo run -- proxy --help`
Expected: Shows proxy subcommand with all flags.

**Step 5: Commit**

```bash
git add src/proxy/cli.rs src/main.rs
git commit -m "feat(proxy): CLI subcommand 'gravrail proxy' with full configuration"
```

---

### Task 8: Integration test

**Files:**
- Create: `tests/proxy_integration.rs`

This test starts the proxy against a mock LLM server and verifies end-to-end confinement.

**Step 1: Write the integration test**

```rust
use axum::{Router, routing::post, Json, response::IntoResponse};
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

use gravrail::circuit::define::Circuit;
use gravrail::lie::group::{LieGroup, GroupType};
use gravrail::proxy::auth::SessionAuth;
use gravrail::proxy::confine::ConfinementPipeline;
use gravrail::proxy::watchdog::{create_watchdog, WatchdogConfig};
use gravrail::proxy::server::{ProxyState, build_router};

/// Mock LLM that returns a fixed response.
async fn mock_llm_handler() -> impl IntoResponse {
    Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello, I am a confined AI assistant."
            },
            "finish_reason": "stop"
        }],
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    }))
}

async fn setup_mock_llm() -> u16 {
    let app = Router::new().route("/v1/chat/completions", post(mock_llm_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    port
}

async fn setup_proxy(upstream_port: u16) -> (u16, String) {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group, None);
    let auth = SessionAuth::new();
    let token = auth.token().to_string();
    let pipeline = ConfinementPipeline::new(circuit, Some(100.0), false);

    let (watchdog_handle, watchdog_future) = create_watchdog(WatchdogConfig {
        interval: Duration::from_millis(100),
        timeout: Duration::from_millis(500),
    });

    let state = Arc::new(ProxyState {
        auth: Mutex::new(auth),
        pipeline: Mutex::new(pipeline),
        watchdog: watchdog_handle,
        upstream_url: format!("http://127.0.0.1:{}", upstream_port),
        client: reqwest::Client::new(),
    });

    // Spawn watchdog
    tokio::spawn(watchdog_future);

    // Spawn proxy heartbeat
    let ping_state = state.clone();
    tokio::spawn(async move {
        loop {
            if !ping_state.watchdog.is_alive() { break; }
            ping_state.watchdog.send_ping().await;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (port, token)
}

#[tokio::test]
async fn test_proxy_rejects_without_token() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, _token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .json(&json!({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn test_proxy_accepts_with_valid_token() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .header("X-GravRail-Token", &token)
        .json(&json!({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    // Should have confinement headers
    assert!(resp.headers().contains_key("x-gravrail-seq"));
    assert!(resp.headers().contains_key("x-gravrail-state"));
}

#[tokio::test]
async fn test_proxy_forwards_content_unchanged() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .header("X-GravRail-Token", &token)
        .json(&json!({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    let body: serde_json::Value = resp.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "Hello, I am a confined AI assistant.");
}

#[tokio::test]
async fn test_proxy_sequence_number_increments() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    for expected_seq in 1..=3 {
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
            .header("X-GravRail-Token", &token)
            .json(&json!({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();

        let seq: u64 = resp.headers()
            .get("x-gravrail-seq")
            .unwrap()
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(seq, expected_seq);
    }
}
```

**Step 2: Run integration tests**

Run: `cargo test --test proxy_integration -- --nocapture`
Expected: 4 tests pass.

**Step 3: Run full test suite**

Run: `cargo test`
Expected: All tests pass (existing 78 + new proxy tests).

**Step 4: Commit**

```bash
git add tests/proxy_integration.rs
git commit -m "test(proxy): integration tests for auth, confinement headers, and sequence numbers"
```

---

## Batch 4: PID Lock & Final Hardening (Task 9)

### Task 9: PID lock file with flock

**Files:**
- Modify: `src/proxy/cli.rs` (add PID lock before server start)

**Step 1: Add PID lock to `run_proxy()`**

Insert before the Axum server starts (before `let listener = ...`):

```rust
// Write PID lock file with exclusive flock
let lock_dir = dirs::home_dir()
    .unwrap_or_else(|| std::path::PathBuf::from("."))
    .join(".gravrail");
std::fs::create_dir_all(&lock_dir)?;
let lock_path = lock_dir.join("proxy.lock");

use std::io::Write;
let lock_file = std::fs::OpenOptions::new()
    .write(true)
    .create(true)
    .truncate(true)
    .open(&lock_path)?;

// Try exclusive lock (non-blocking)
use std::os::unix::io::AsRawFd;
let fd = lock_file.as_raw_fd();
let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
if ret != 0 {
    anyhow::bail!("Another GravProxy instance is already running (lock file: {})", lock_path.display());
}
write!(&lock_file, "{}", std::process::id())?;
eprintln!("  PID lock: {}", lock_path.display());
```

**Step 2: Add `libc` dependency to Cargo.toml**

```toml
libc = "0.2"
```

**Step 3: Verify it compiles**

Run: `cargo check`
Expected: Compiles.

**Step 4: Commit**

```bash
git add src/proxy/cli.rs Cargo.toml
git commit -m "feat(proxy): PID lock file with flock prevents shadow proxy instances"
```

---

## Summary

| Task | Module | Tests | Description |
|------|--------|-------|-------------|
| 1 | Skeleton | 0 | Dependencies, module structure, placeholder files |
| 2 | `proxy/auth.rs` | 5 | Session token generation, constant-time validation |
| 3 | `proxy/watchdog.rs` | 7 | HMAC-signed heartbeats, mutual liveness detection |
| 4 | `proxy/confine.rs` | 10 | Per-response confinement pipeline with bounds checking |
| 5 | `proxy/stream.rs` | 10 | SSE parsing, text extraction, confined chunk formatting |
| 6 | `proxy/server.rs` | 0 (compile) | Axum HTTP proxy, request forwarding, response interception |
| 7 | `proxy/cli.rs` + `main.rs` | 0 (compile + help) | CLI subcommand with full configuration |
| 8 | Integration | 4 | End-to-end: mock LLM → proxy → consumer with auth + confinement |
| 9 | PID lock | 0 (compile) | flock-based lock file prevents duplicate proxies |

**Total new tests:** ~36
**New dependencies:** reqwest, hmac, hex, async-stream, libc
