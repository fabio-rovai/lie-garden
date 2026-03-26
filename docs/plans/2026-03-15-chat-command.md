# `gravrail chat` Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `gravrail chat` CLI subcommand that starts the confinement proxy internally and provides an interactive multi-turn conversation loop where every LLM response is geometrically confined and the confinement metadata is shown inline.

**Architecture:** A new `src/chat/` module hosts the chat loop. It reuses `run_proxy_returning_token` to start the proxy on a random free port, then sends each user message as a streaming POST to the proxy, reads the SSE response line-by-line, and prints both the assistant text and the `[✓ confined | seq=N | state=[...]]` footer after each turn. No new terminal dependencies — plain `stdin().lines()` is sufficient for MVP.

**Tech Stack:** Rust, tokio, reqwest (already present with stream feature), serde_json, clap derive (already present), existing `proxy::cli::run_proxy_returning_token`.

---

## Task 1: Add `chat` subcommand to main.rs

**Files:**
- Modify: `src/main.rs`

**Step 1: Read current main.rs**

Read `/Users/fabio/projects/gravrail/src/main.rs` to understand the current `Commands` enum and how `proxy` is wired.

**Step 2: Add Chat variant to Commands enum**

Inside the existing `Commands` enum, add:

```rust
/// Interactive chat session with a confined LLM.
Chat {
    /// Upstream LLM URL (OpenAI-compatible, e.g. http://localhost:11434/v1)
    #[arg(long, default_value = "http://localhost:11434/v1")]
    upstream: String,

    /// Model name to use
    #[arg(long, default_value = "llama3")]
    model: String,

    /// Lie group type: SO, SE, or GL
    #[arg(long, default_value = "SO")]
    group_type: String,

    /// Group dimension
    #[arg(long, default_value_t = 3)]
    group_dim: usize,

    /// Maximum state norm for reachability (None = unlimited)
    #[arg(long)]
    max_state_norm: Option<f64>,

    /// Generate STARK proof for every response
    #[arg(long, default_value_t = false)]
    prove: bool,
},
```

**Step 3: Add match arm in main()**

In the `match cli.command` block, add:

```rust
Commands::Chat { upstream, model, group_type, group_dim, max_state_norm, prove } => {
    gravrail::chat::run_chat(gravrail::chat::ChatConfig {
        upstream_url: upstream,
        model,
        group_type,
        group_dim,
        max_state_norm,
        prove_every_step: prove,
    })
    .await?;
}
```

**Step 4: Add `pub mod chat;` to lib.rs**

Read `src/lib.rs`. Add `pub mod chat;` to the module list.

**Step 5: Verify it compiles (will fail — chat module doesn't exist yet)**

Run: `/Users/fabio/.cargo/bin/cargo check 2>&1 | tail -20`
Expected: `error[E0583]: file not found for module 'chat'`

**Step 6: Commit the stub**

```bash
cd /Users/fabio/projects/gravrail && git add src/main.rs src/lib.rs && git commit -m "feat(chat): add chat subcommand stub to CLI"
```

---

## Task 2: Create src/chat/mod.rs — core chat loop

**Files:**
- Create: `src/chat/mod.rs`

**Step 1: Write the module**

Create `/Users/fabio/projects/gravrail/src/chat/mod.rs`:

```rust
//! Interactive chat loop with transport-level confinement.
//!
//! Starts GravProxy on a free local port, then runs a read-eval-print loop
//! that sends each user message to the proxy and displays the assistant reply
//! with confinement metadata.

use crate::proxy::cli::{ProxyConfig, run_proxy_returning_token};
use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::oneshot;

/// Configuration for the chat session.
pub struct ChatConfig {
    pub upstream_url: String,
    pub model: String,
    pub group_type: String,
    pub group_dim: usize,
    pub max_state_norm: Option<f64>,
    pub prove_every_step: bool,
}

/// Run the interactive chat loop.
pub async fn run_chat(config: ChatConfig) -> Result<()> {
    // 1. Find a free port
    let port = free_port();

    // 2. Start proxy in background, capture token
    let (token_tx, token_rx) = oneshot::channel::<String>();
    let mut token_tx = Some(token_tx);

    let proxy_config = ProxyConfig {
        circuit_id: None,
        group_type: config.group_type.clone(),
        group_dim: config.group_dim,
        upstream_url: config.upstream_url.clone(),
        port,
        heartbeat_interval_ms: 500,
        heartbeat_timeout_ms: 2000,
        max_state_norm: config.max_state_norm,
        prove_every_step: config.prove_every_step,
    };

    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);

    tokio::spawn(async move {
        run_proxy_returning_token(proxy_config, move |t| {
            if let Some(tx) = token_tx.take() {
                let _ = tx.send(t);
            }
        })
        .await
        .ok();
        running_clone.store(false, Ordering::SeqCst);
    });

    // 3. Wait for token
    let token = tokio::time::timeout(Duration::from_secs(5), token_rx)
        .await
        .map_err(|_| anyhow::anyhow!("proxy did not start within 5 seconds"))?
        .map_err(|_| anyhow::anyhow!("proxy token channel closed"))?;

    // Small delay to let the proxy bind and start serving
    tokio::time::sleep(Duration::from_millis(200)).await;

    // 4. Print banner
    eprintln!(
        "[gravrail] circuit: {}({}) | upstream: {}",
        config.group_type, config.group_dim, config.upstream_url
    );
    if let Some(norm) = config.max_state_norm {
        eprintln!("[gravrail] max state norm: {}", norm);
    }
    eprintln!("[gravrail] type 'exit' or press Ctrl-C to quit\n");

    // 5. Build HTTP client
    let client = Client::new();
    let proxy_url = format!("http://127.0.0.1:{}", port);

    // 6. Conversation history
    let mut messages: Vec<Value> = Vec::new();

    // 7. Read-eval-print loop
    let stdin = std::io::stdin();
    loop {
        // Check proxy still alive
        if !running.load(Ordering::SeqCst) {
            eprintln!("[gravrail] proxy died — exiting");
            break;
        }

        // Print prompt
        print!("You: ");
        std::io::stdout().flush().ok();

        // Read line (blocking — offload to spawn_blocking)
        let line = {
            let mut line = String::new();
            match stdin.lock().read_line(&mut line) {
                Ok(0) => break, // EOF
                Ok(_) => line.trim().to_string(),
                Err(_) => break,
            }
        };

        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        // Add user message to history
        messages.push(json!({"role": "user", "content": line}));

        // Send streaming request to proxy
        let body = json!({
            "model": config.model,
            "messages": messages,
            "stream": true,
        });

        let resp = match client
            .post(format!("{}/v1/chat/completions", proxy_url))
            .header("x-gravrail-token", &token)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[gravrail] request error: {}", e);
                continue;
            }
        };

        let status = resp.status();
        if status == 403 {
            eprintln!("[gravrail] ✗ BLOCKED: reachability violation");
            // Remove last user message since we got no reply
            messages.pop();
            continue;
        }
        if !status.is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            eprintln!("[gravrail] upstream error {}: {}", status, body_text);
            messages.pop();
            continue;
        }

        // Capture confinement metadata from response headers
        let seq = resp
            .headers()
            .get("x-gravrail-seq")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("?")
            .to_string();
        let state = resp
            .headers()
            .get("x-gravrail-state")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("?")
            .to_string();

        // Stream the response body
        print!("Assistant: ");
        std::io::stdout().flush().ok();

        let mut assistant_text = String::new();
        let body_text = resp.text().await.unwrap_or_default();

        for line in body_text.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" {
                    break;
                }
                if let Ok(json) = serde_json::from_str::<Value>(data) {
                    // Try streaming delta
                    if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                        assistant_text.push_str(content);
                    }
                    // Try non-streaming message
                    if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                        assistant_text.push_str(content);
                    }
                }
            }
        }

        // For non-streaming responses (the proxy may buffer the whole thing):
        // If assistant_text is still empty, try parsing as regular JSON
        if assistant_text.is_empty() {
            if let Ok(json) = serde_json::from_str::<Value>(&body_text) {
                if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                    print!("{}", content);
                    std::io::stdout().flush().ok();
                    assistant_text.push_str(content);
                }
            }
        }

        println!();
        println!("[✓ confined | seq={} | state={}]", seq, state);
        println!();

        // Add assistant reply to history
        if !assistant_text.is_empty() {
            messages.push(json!({"role": "assistant", "content": assistant_text}));
        }
    }

    println!("[gravrail] session ended");
    Ok(())
}

/// Find a free TCP port by binding to port 0.
fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("failed to bind ephemeral port")
        .local_addr()
        .unwrap()
        .port()
}
```

**Step 2: Verify it compiles**

Run: `/Users/fabio/.cargo/bin/cargo check 2>&1 | tail -30`
Expected: Clean compile (0 errors)

If there are errors, fix them. Common issues:
- `run_proxy_returning_token` not exported from `proxy` module — check `src/proxy/mod.rs`
- Import path for `ProxyConfig` wrong — it's `crate::proxy::cli::ProxyConfig`

**Step 3: Build the binary**

Run: `/Users/fabio/.cargo/bin/cargo build 2>&1 | tail -20`
Expected: `Finished dev` with 0 errors

**Step 4: Smoke test — verify help text**

Run: `./target/debug/gravrail chat --help 2>&1`
Expected: Shows chat subcommand with `--upstream`, `--model`, `--group-type`, `--group-dim`, `--max-state-norm`, `--prove` flags

**Step 5: Commit**

```bash
cd /Users/fabio/projects/gravrail && git add src/chat/mod.rs src/main.rs src/lib.rs && git commit -m "feat(chat): add gravrail chat command with inline confinement metadata"
```

---

## Task 3: Test the chat module

**Files:**
- Create: `tests/chat_test.rs`

**Step 1: Write a test that exercises the free_port helper and proxy startup**

The chat loop is interactive (reads stdin) so we can't test the full loop easily. Instead, test the building blocks:

```rust
//! Tests for the chat module infrastructure.

use gravrail::proxy::cli::{ProxyConfig, run_proxy_returning_token};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn test_chat_proxy_startup_and_request() {
    // Simulate what run_chat does: start proxy against mock LLM, send one message
    let mock_server = MockServer::start().await;

    // Mock streaming response (what an LLM returns for a chat request)
    let sse_body = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" there!\"},\"index\":0}]}\n\n",
        "data: [DONE]\n\n",
    );

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .mount(&mock_server)
        .await;

    // Start proxy
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    let mut tx = Some(tx);

    tokio::spawn(async move {
        run_proxy_returning_token(
            ProxyConfig {
                circuit_id: None,
                group_type: "SO".to_string(),
                group_dim: 3,
                upstream_url: mock_server.uri(),
                port,
                heartbeat_interval_ms: 500,
                heartbeat_timeout_ms: 2000,
                max_state_norm: None,
                prove_every_step: false,
            },
            move |t| { if let Some(s) = tx.take() { let _ = s.send(t); } },
        )
        .await
        .ok();
    });

    let token = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .expect("timeout")
        .expect("channel");
    sleep(Duration::from_millis(200)).await;

    // Send a chat-style request through the proxy
    let client = Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);

    // Confinement headers should be present
    assert!(resp.headers().contains_key("x-gravrail-seq"));
    assert!(resp.headers().contains_key("x-gravrail-state"));

    let seq: u64 = resp.headers()["x-gravrail-seq"]
        .to_str().unwrap().parse().unwrap();
    assert_eq!(seq, 1);

    // Body should contain SSE data
    let body = resp.text().await.unwrap();
    assert!(body.contains("data:"), "body: {}", body);
}

#[tokio::test]
async fn test_chat_multi_turn_sequence_increments() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": "reply"}, "index": 0}]
        })))
        .expect(2)  // Two requests
        .mount(&mock_server)
        .await;

    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    let mut tx = Some(tx);

    tokio::spawn(async move {
        run_proxy_returning_token(
            ProxyConfig {
                circuit_id: None,
                group_type: "SO".to_string(),
                group_dim: 3,
                upstream_url: mock_server.uri(),
                port,
                heartbeat_interval_ms: 500,
                heartbeat_timeout_ms: 2000,
                max_state_norm: None,
                prove_every_step: false,
            },
            move |t| { if let Some(s) = tx.take() { let _ = s.send(t); } },
        )
        .await
        .ok();
    });

    let token = tokio::time::timeout(Duration::from_secs(5), rx)
        .await.unwrap().unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    let client = Client::new();

    // First turn
    let r1 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "llama3", "messages": [{"role": "user", "content": "turn 1"}]}))
        .send().await.unwrap();
    let seq1: u64 = r1.headers()["x-gravrail-seq"].to_str().unwrap().parse().unwrap();

    // Second turn
    let r2 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "llama3", "messages": [
            {"role": "user", "content": "turn 1"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "turn 2"},
        ]}))
        .send().await.unwrap();
    let seq2: u64 = r2.headers()["x-gravrail-seq"].to_str().unwrap().parse().unwrap();

    assert_eq!(seq1, 1, "first turn seq should be 1");
    assert_eq!(seq2, 2, "second turn seq should be 2");
}
```

**Step 2: Run tests**

Run: `/Users/fabio/.cargo/bin/cargo test --test chat_test -- --nocapture 2>&1 | tail -30`
Expected: 2 tests PASS

**Step 3: Run full suite**

Run: `/Users/fabio/.cargo/bin/cargo test 2>&1 | grep -E "^test result|FAILED" | tail -20`
Expected: All PASS, 0 failures

**Step 4: Commit**

```bash
cd /Users/fabio/projects/gravrail && git add tests/chat_test.rs && git commit -m "test: chat session infrastructure tests (proxy startup, multi-turn seq)"
```

---

## Task 4: Manual smoke test and push

**Step 1: Build release binary**

Run: `/Users/fabio/.cargo/bin/cargo build --release 2>&1 | tail -10`
Expected: `Finished release`

**Step 2: Verify help text**

Run: `./target/release/gravrail chat --help`
Expected: Full help with all flags

**Step 3: (Optional, if Ollama is available) Live test**

If Ollama is running locally:
```bash
./target/release/gravrail chat --upstream http://localhost:11434/v1 --model llama3
```
Type a message, verify `[✓ confined | seq=1 | state=[...]]` appears after each reply.

If Ollama is not available, skip this step.

**Step 4: Push to GitHub**

```bash
cd /Users/fabio/projects/gravrail && git push
```

---

## Expected UX

```
$ gravrail chat --upstream http://localhost:11434/v1 --model llama3
[gravrail] circuit: SO(3) | upstream: http://localhost:11434/v1
[gravrail] type 'exit' or press Ctrl-C to quit

You: What is the capital of France?
Assistant: The capital of France is Paris.
[✓ confined | seq=1 | state=[0.998,0.043,-0.021,...]]

You: And Germany?
Assistant: The capital of Germany is Berlin.
[✓ confined | seq=2 | state=[0.995,0.087,-0.015,...]]

You: exit
[gravrail] session ended
```
