// Integration tests for GravProxy: auth, confinement headers, sequence numbers.

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

/// Mock LLM handler returning a fixed OpenAI chat completion response.
async fn mock_llm_handler() -> impl IntoResponse {
    Json(json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello, I am a confined AI assistant."}, "finish_reason": "stop"}],
        "model": "mock-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    }))
}

/// Start a mock LLM server on a random port, return the port.
async fn setup_mock_llm() -> u16 {
    let app = Router::new().route("/v1/chat/completions", post(mock_llm_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    port
}

/// Create the full proxy stack pointing at the given upstream port.
/// Returns (proxy_port, session_token).
async fn setup_proxy(upstream_port: u16) -> (u16, String) {
    // SO(3) circuit, no generator mask
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group, None);

    // Session auth — grab the token before wrapping in Mutex
    let auth = SessionAuth::new();
    let token = auth.token().to_owned();

    // Confinement pipeline: generous bound, no per-step proofs
    let pipeline = ConfinementPipeline::new(circuit, Some(100.0), false);

    // Watchdog with short intervals for tests
    let config = WatchdogConfig {
        interval: Duration::from_millis(100),
        timeout: Duration::from_millis(500),
    };
    let (watchdog_handle, watchdog_loop) = create_watchdog(config);

    // Spawn the watchdog loop
    tokio::spawn(watchdog_loop);

    // Spawn a heartbeat sender that keeps the watchdog alive
    let ping_handle_killed = watchdog_handle.killed.clone();
    let ping_tx = watchdog_handle.ping_tx.clone();
    let hmac_key = watchdog_handle.hmac_key().to_vec();
    tokio::spawn(async move {
        loop {
            if ping_handle_killed.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            let hb = gravrail::proxy::watchdog::sign_heartbeat(0, &hmac_key);
            if ping_tx.send(hb).await.is_err() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    });

    // Build proxy state
    let state = Arc::new(ProxyState {
        auth: Mutex::new(auth),
        pipeline: Mutex::new(pipeline),
        watchdog: watchdog_handle,
        upstream_url: format!("http://127.0.0.1:{}", upstream_port),
        client: reqwest::Client::new(),
    });

    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    (proxy_port, token)
}

#[tokio::test]
async fn test_proxy_rejects_without_token() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, _token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .json(&json!({"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 401);
}

#[tokio::test]
async fn test_proxy_accepts_with_valid_token() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
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
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status().as_u16(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "Hello, I am a confined AI assistant.");
}

#[tokio::test]
async fn test_proxy_sequence_number_increments() {
    let llm_port = setup_mock_llm().await;
    let (proxy_port, token) = setup_proxy(llm_port).await;

    let client = reqwest::Client::new();
    let mut seq_values = Vec::new();

    for _ in 0..3 {
        let resp = client
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", proxy_port))
            .header("x-gravrail-token", &token)
            .json(&json!({"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status().as_u16(), 200);
        let seq_str = resp
            .headers()
            .get("x-gravrail-seq")
            .unwrap()
            .to_str()
            .unwrap();
        let seq: u64 = seq_str.parse().unwrap();
        seq_values.push(seq);
    }

    assert_eq!(seq_values, vec![1, 2, 3]);
}
