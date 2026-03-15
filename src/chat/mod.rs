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
        holonomy_window_size: 8,
        json_startup: false,
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

        // Read line
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

        // Send non-streaming request so confinement headers are in HTTP response headers.
        // (Streaming puts metadata in SSE comment lines; non-streaming uses x-gravrail-* headers.)
        let body = json!({
            "model": config.model,
            "messages": messages,
            "stream": false,
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

        // Read body and stream to stdout
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
                    if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                        assistant_text.push_str(content);
                    }
                    if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                        print!("{}", content);
                        std::io::stdout().flush().ok();
                        assistant_text.push_str(content);
                    }
                }
            }
        }

        // Fallback: try parsing whole body as non-streaming JSON
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
