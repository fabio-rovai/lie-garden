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
