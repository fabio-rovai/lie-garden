//! Integration tests for the chat command infrastructure.
//! Tests the proxy behavior that run_chat depends on.

use gravrail::proxy::cli::{ProxyConfig, run_proxy_returning_token};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn start_proxy(mock_url: String) -> (u16, String) {
    let port = free_port();
    let (tx, rx) = tokio::sync::oneshot::channel::<String>();
    let mut tx = Some(tx);
    tokio::spawn(async move {
        run_proxy_returning_token(
            ProxyConfig {
                circuit_id: None,
                group_type: "SO".to_string(),
                group_dim: 3,
                upstream_url: mock_url,
                port,
                heartbeat_interval_ms: 500,
                heartbeat_timeout_ms: 2000,
                max_state_norm: None,
                prove_every_step: false,
                holonomy_window_size: 8,
                json_startup: false,
                holonomy_threshold: 0.0,
                input_holonomy_threshold: 0.0,
                input_norm_threshold: 0.0,
            },
            move |t| { if let Some(s) = tx.take() { let _ = s.send(t); } },
        )
        .await
        .ok();
    });
    let token = tokio::time::timeout(Duration::from_secs(5), rx)
        .await.expect("timeout").expect("channel");
    sleep(Duration::from_millis(200)).await;
    (port, token)
}

#[tokio::test]
async fn test_chat_proxy_streaming_request() {
    // Simulate what run_chat does: send a streaming chat request through the proxy.
    // For streaming responses, the proxy embeds confinement metadata as SSE comments
    // in the body (": x-gravrail-seq=N") rather than as HTTP response headers.
    let mock_server = MockServer::start().await;

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

    let (port, token) = start_proxy(mock_server.uri()).await;

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

    // For streaming, the proxy returns text/event-stream
    let content_type = resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.contains("text/event-stream"), "expected text/event-stream, got: {}", content_type);

    let body = resp.text().await.unwrap();

    // Confinement metadata is injected as SSE comments in the body
    assert!(body.contains(": x-gravrail-seq="), "body should contain SSE seq comment: {}", body);
    assert!(body.contains(": x-gravrail-state="), "body should contain SSE state comment: {}", body);

    // Verify seq starts at 1 for the first chunk with content
    assert!(body.contains(": x-gravrail-seq=1"), "first SSE chunk should have seq=1: {}", body);

    // Original SSE data and [DONE] should pass through
    assert!(body.contains("data:"), "body should contain SSE data lines: {}", body);
    assert!(body.contains("[DONE]"), "body should contain [DONE]: {}", body);
}

#[tokio::test]
async fn test_chat_multi_turn_seq_increments() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": "reply"}, "index": 0, "finish_reason": "stop"}]
        })))
        .mount(&mock_server)
        .await;

    let (port, token) = start_proxy(mock_server.uri()).await;
    let client = Client::new();

    // Turn 1
    let r1 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "llama3", "messages": [{"role": "user", "content": "turn 1"}]}))
        .send().await.unwrap();
    let seq1: u64 = r1.headers()["x-gravrail-seq"].to_str().unwrap().parse().unwrap();

    // Turn 2 (with history)
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

    assert_eq!(seq1, 1, "first turn should have seq=1");
    assert_eq!(seq2, 2, "second turn should have seq=2");
}

#[tokio::test]
async fn test_chat_state_changes_across_turns() {
    // The confinement state should evolve with each message
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"message": {"role": "assistant", "content": "hello world response"}, "index": 0}]
        })))
        .mount(&mock_server)
        .await;

    let (port, token) = start_proxy(mock_server.uri()).await;
    let client = Client::new();

    let r1 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "llama3", "messages": [{"role": "user", "content": "msg1"}]}))
        .send().await.unwrap();
    let state1 = r1.headers()["x-gravrail-state"].to_str().unwrap().to_string();

    let r2 = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({"model": "llama3", "messages": [{"role": "user", "content": "msg2"}]}))
        .send().await.unwrap();
    let state2 = r2.headers()["x-gravrail-state"].to_str().unwrap().to_string();

    // State should change between turns (the Lie group element evolves)
    assert_ne!(state1, state2, "confinement state should evolve across turns");
}
