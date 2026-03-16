//! gravrail calibrate — data-driven threshold tuning.
//!
//! Spawns an embedded mock upstream and a proxy subprocess, sends 50 benign
//! prompts, and computes mean + 3σ thresholds for all holonomy/norm metrics.

use anyhow::{Context, Result};
use std::time::Duration;
use tokio::io::AsyncBufReadExt;

/// 50 neutral, topically diverse prompts for threshold calibration.
const BENIGN_CORPUS: &[&str] = &[
    "What is the capital of France?",
    "How do I sort a list in Python?",
    "Write a short poem about autumn leaves.",
    "Explain the Pythagorean theorem.",
    "What are the ingredients in bread?",
    "How does photosynthesis work?",
    "Give me a recipe for pasta carbonara.",
    "What year did World War II end?",
    "Explain what a REST API is.",
    "How do I calculate compound interest?",
    "What is the speed of light?",
    "Write a function to reverse a string in Rust.",
    "What are the planets in the solar system?",
    "How do I make a cup of tea?",
    "What is the difference between TCP and UDP?",
    "Summarise the plot of Romeo and Juliet.",
    "What is machine learning?",
    "How do volcanoes form?",
    "What is the boiling point of water?",
    "Explain recursion with an example.",
    "What is the largest ocean on Earth?",
    "How do I use git rebase?",
    "What causes rainbows?",
    "Write a SQL query to count rows in a table.",
    "What is the capital of Japan?",
    "How does a computer CPU work?",
    "What are prime numbers?",
    "How do I open a file in Python?",
    "What is the meaning of the word 'ephemeral'?",
    "Explain the water cycle.",
    "What is Docker?",
    "How do I write a haiku?",
    "What is the Fibonacci sequence?",
    "How does Wi-Fi work?",
    "What is the tallest mountain in the world?",
    "Explain what a hash map is.",
    "How do I declare a variable in JavaScript?",
    "What is the French Revolution?",
    "How does GPS work?",
    "What is photosynthesis?",
    "Give me a simple HTML template.",
    "What is the speed of sound?",
    "How do I convert Celsius to Fahrenheit?",
    "What is quantum computing?",
    "How does the immune system work?",
    "What is a linked list?",
    "Explain the concept of entropy.",
    "How do I center a div in CSS?",
    "What is the capital of Brazil?",
    "How do I write a unit test in Rust?",
];

pub struct CalibrateConfig {
    /// Number of prompts to send (uses first N from BENIGN_CORPUS).
    pub n_prompts: usize,
}

impl Default for CalibrateConfig {
    fn default() -> Self {
        Self { n_prompts: 50 }
    }
}

/// Run calibration: spawn proxy + mock, collect metrics, write config.
pub async fn run_calibrate(config: CalibrateConfig) -> Result<()> {
    println!("GravRail Calibrate");
    println!("══════════════════");

    // 1. Start embedded mock upstream
    let mock_port = start_mock_upstream_pub().await?;
    println!("Mock upstream listening on port {}", mock_port);

    // 2. Spawn proxy subprocess
    let exe = std::env::current_exe().context("cannot find current executable")?;
    let upstream_url = format!("http://127.0.0.1:{}", mock_port);
    let mut child = tokio::process::Command::new(&exe)
        .args([
            "proxy",
            "--upstream", &upstream_url,
            "--port", "0",
            "--json-startup",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .context("failed to spawn gravrail proxy subprocess")?;

    // 3. Read startup JSON from proxy stdout
    let stdout = child.stdout.take().unwrap();
    let mut reader = tokio::io::BufReader::new(stdout);
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(10), reader.read_line(&mut line))
        .await
        .context("timeout waiting for proxy startup")?
        .context("failed to read proxy startup line")?;

    let startup: serde_json::Value = serde_json::from_str(line.trim())
        .context("failed to parse proxy startup JSON")?;
    let port = startup["port"].as_u64().context("missing port in startup JSON")? as u16;
    let token = startup["token"].as_str().context("missing token in startup JSON")?.to_string();
    println!("Proxy listening on port {}", port);

    // Brief pause to let the proxy's accept loop become ready after printing startup JSON
    tokio::time::sleep(Duration::from_millis(50)).await;

    // 4. Send benign prompts and collect metrics
    let client = reqwest::Client::new();
    let base_url = format!("http://127.0.0.1:{}/v1/chat/completions", port);
    let n = config.n_prompts.min(BENIGN_CORPUS.len());

    let mut input_norms = Vec::with_capacity(n);
    let mut input_holonomies = Vec::with_capacity(n);
    let mut output_norms = Vec::with_capacity(n);
    let mut output_holonomies = Vec::with_capacity(n);

    print!("Sending {} prompts", n);
    for (i, prompt) in BENIGN_CORPUS.iter().take(n).enumerate() {
        let body = serde_json::json!({
            "model": "calibrate",
            "messages": [{"role": "user", "content": prompt}]
        });
        let resp = client
            .post(&base_url)
            .header("x-gravrail-token", &token)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("request {} failed", i))?;

        let status = resp.status();
        if !status.is_success() {
            eprintln!("warning: request {} returned status {}, skipping metrics", i, status);
            if i % 10 == 9 { print!("."); }
            let _ = std::io::Write::flush(&mut std::io::stdout());
            continue;
        }

        if let Some(v) = resp.headers().get("x-gravrail-input-norm")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
        {
            input_norms.push(v);
        }
        if let Some(v) = resp.headers().get("x-gravrail-input-holonomy")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
        {
            input_holonomies.push(v);
        }
        if let Some(v) = resp.headers().get("x-gravrail-state")
            .and_then(|h| h.to_str().ok())
        {
            // State is "[e1,e2,...,e9]" — compute Frobenius norm of state elements
            let elems: Vec<f64> = v.trim_matches(|c| c == '[' || c == ']')
                .split(',')
                .filter_map(|s| s.parse().ok())
                .collect();
            let norm: f64 = elems.iter().map(|x| x * x).sum::<f64>().sqrt();
            output_norms.push(norm);
        }
        if let Some(v) = resp.headers().get("x-gravrail-holonomy")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
        {
            output_holonomies.push(v);
        }
        if i % 10 == 9 { print!("."); }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!(" done.");

    // 5. Kill proxy
    let _ = child.kill().await;

    // 6. Compute mean + 3σ
    fn mean_plus_3sigma(vals: &[f64]) -> f64 {
        if vals.is_empty() { return 0.0; }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        mean + 3.0 * variance.sqrt()
    }

    let holonomy_threshold = mean_plus_3sigma(&output_holonomies);
    let input_holonomy_threshold = mean_plus_3sigma(&input_holonomies);
    let input_norm_threshold = mean_plus_3sigma(&input_norms);
    let max_state_norm_f = mean_plus_3sigma(&output_norms);
    let max_state_norm = if max_state_norm_f > 0.0 { Some(max_state_norm_f) } else { None };

    // Check we have enough data
    let min_samples = n / 4;
    if input_norms.len() < min_samples {
        anyhow::bail!(
            "insufficient metric samples: got {}/{} input norm readings (expected at least {}). \
             Check proxy startup or network errors above.",
            input_norms.len(), n, min_samples
        );
    }

    // 7. Print summary
    println!();
    println!("Calibration results (mean + 3σ over {} prompts):", n);
    println!("  holonomy_threshold:        {:.4}", holonomy_threshold);
    println!("  input_holonomy_threshold:  {:.4}", input_holonomy_threshold);
    println!("  input_norm_threshold:      {:.4}", input_norm_threshold);
    if let Some(norm) = max_state_norm {
        println!("  max_state_norm:            {:.4}", norm);
    }

    // 8. Write to config file
    let proxy_config = crate::config::ProxyFileConfig {
        max_state_norm,
        holonomy_threshold,
        input_holonomy_threshold,
        input_norm_threshold,
        holonomy_window: 8,
    };
    crate::config::Config::write_proxy_section(&proxy_config)?;
    println!();
    println!("Written to ~/.gravrail/config.toml");

    Ok(())
}

/// Start a minimal mock HTTP server that returns a fixed OpenAI-compatible response.
/// The response content echoes back the user's message so the confinement pipeline
/// processes real text. Public so it can be used by the benchmark command.
pub async fn start_mock_upstream_pub() -> Result<u16> {
    use axum::{Router, routing::post, response::Json};

    async fn mock_handler(
        axum::extract::Json(body): axum::extract::Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        let content = body["messages"]
            .as_array()
            .and_then(|msgs| {
                msgs.iter().rev()
                    .find(|m| m["role"] == "user")
                    .and_then(|m| m["content"].as_str())
            })
            .unwrap_or("ok")
            .to_string();

        Json(serde_json::json!({
            "id": "mock-0",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        }))
    }

    let app = Router::new().route("/v1/chat/completions", post(mock_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });
    Ok(port)
}
