//! End-to-end integration tests for GravProxy.
//! Uses wiremock to simulate an OpenAI-compatible LLM without needing a live server.

use gravrail::proxy::cli::{ProxyConfig, run_proxy_returning_token};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;
use tokio::time::sleep;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn get_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

/// Spawn proxy pointing at mock_url, return (port, session_token).
async fn start_proxy_against(mock_url: String) -> (u16, String) {
    let port = get_free_port();
    let (token_tx, token_rx) = tokio::sync::oneshot::channel::<String>();
    let mut token_tx = Some(token_tx);

    let config = ProxyConfig {
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
    };

    tokio::spawn(async move {
        run_proxy_returning_token(config, move |t| {
            if let Some(tx) = token_tx.take() {
                let _ = tx.send(t);
            }
        })
        .await
        .ok();
    });

    let token = token_rx.await.expect("token callback should fire");
    // Give the proxy a moment to bind and start serving
    sleep(Duration::from_millis(150)).await;
    (port, token)
}

#[tokio::test]
async fn test_proxy_non_streaming_response_confined() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from mock LLM"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        })))
        .mount(&mock_server)
        .await;

    let (port, token) = start_proxy_against(mock_server.uri()).await;

    let client = Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers().contains_key("x-gravrail-seq"),
        "missing x-gravrail-seq header"
    );
    assert!(
        resp.headers().contains_key("x-gravrail-state"),
        "missing x-gravrail-state header"
    );

    let seq: u64 = resp.headers()["x-gravrail-seq"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(seq, 1, "first response should have seq=1");
}

#[tokio::test]
async fn test_proxy_streaming_response_confined() {
    let mock_server = MockServer::start().await;

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n",
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

    let (port, token) = start_proxy_against(mock_server.uri()).await;

    let client = Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let content_type = resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(content_type.contains("text/event-stream"), "got content-type: {}", content_type);

    let body = resp.text().await.unwrap();
    assert!(body.contains("data:"), "body should contain SSE data lines: {}", body);
    // [DONE] passes through unchanged
    assert!(body.contains("[DONE]"), "body should contain [DONE]: {}", body);
}

#[tokio::test]
async fn test_proxy_blocks_reachability_violation() {
    let mock_server = MockServer::start().await;

    // Respond with very large text content
    let large_text = "x".repeat(50_000);
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-violation",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": large_text},
                "finish_reason": "stop"
            }]
        })))
        .mount(&mock_server)
        .await;

    // Start a proxy with an extremely tight reachability bound
    let port = get_free_port();
    let (token_tx, token_rx) = tokio::sync::oneshot::channel::<String>();
    let mut token_tx = Some(token_tx);
    let mock_uri = mock_server.uri();

    tokio::spawn(async move {
        run_proxy_returning_token(
            ProxyConfig {
                circuit_id: None,
                group_type: "SO".to_string(),
                group_dim: 3,
                upstream_url: mock_uri,
                port,
                heartbeat_interval_ms: 500,
                heartbeat_timeout_ms: 2000,
                max_state_norm: Some(0.001), // extremely tight bound
                prove_every_step: false,
                holonomy_window_size: 8,
                json_startup: false,
            },
            move |t| {
                if let Some(tx) = token_tx.take() {
                    let _ = tx.send(t);
                }
            },
        )
        .await
        .ok();
    });

    let token = token_rx.await.expect("token should be received");
    sleep(Duration::from_millis(150)).await;

    let client = Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", &token)
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        403,
        "expected 403 for reachability violation, got {}",
        resp.status()
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    let error = body["error"].as_str().unwrap_or("");
    assert!(
        error.contains("reachability_violation"),
        "expected reachability_violation error, got: {}",
        error
    );
}

#[tokio::test]
async fn test_proxy_rejects_invalid_token() {
    let mock_server = MockServer::start().await;
    // No mock needed — auth check happens before upstream call

    let (port, _real_token) = start_proxy_against(mock_server.uri()).await;

    let client = Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("x-gravrail-token", "totally-wrong-token-xxxx")
        .json(&json!({"model": "gpt-4", "messages": []}))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 401, "expected 401, got {}", resp.status());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["error"], "invalid_token");
}
