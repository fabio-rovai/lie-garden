// Axum HTTP server, request forwarding, response interception

use axum::{
    Router,
    body::{Body, Bytes},
    extract::State,
    http::{Request, Response, StatusCode, HeaderMap},
    response::IntoResponse,
    routing::any,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use reqwest::Client;
use futures_util::StreamExt;

use super::auth::SessionAuth;
use super::confine::{ConfinementPipeline, ConfinementError};
use super::watchdog::WatchdogHandle;
use super::stream::{extract_text_content, parse_sse_data, format_sse_chunk};

/// Maximum request body size: 10 MB.
const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

/// Shared proxy state threaded through every handler.
pub struct ProxyState {
    pub auth: Mutex<SessionAuth>,
    pub pipeline: Mutex<ConfinementPipeline>,
    pub watchdog: WatchdogHandle,
    pub upstream_url: String,
    pub client: Client,
}

/// Build the Axum router with a catch-all handler.
pub fn build_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/{*path}", any(proxy_handler))
        .with_state(state)
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_SIZE))
}

/// Main proxy handler for every incoming request.
async fn proxy_handler(
    State(state): State<Arc<ProxyState>>,
    req: Request<Body>,
) -> impl IntoResponse {
    // 1. Check watchdog liveness
    if !state.watchdog.is_alive() {
        return json_error_response(StatusCode::SERVICE_UNAVAILABLE, "proxy_killed");
    }

    // 2. Validate session token
    {
        let mut auth = state.auth.lock().await;
        let token = req
            .headers()
            .get("x-gravrail-token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !auth.validate(token) {
            return json_error_response(StatusCode::UNAUTHORIZED, "invalid_token");
        }
    }

    // 3. Build upstream request
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(|q| q.to_string());

    // Copy headers, stripping host and token
    let mut upstream_headers = HeaderMap::new();
    for (name, value) in req.headers().iter() {
        let key = name.as_str().to_lowercase();
        if key == "host" || key == "x-gravrail-token" {
            continue;
        }
        upstream_headers.insert(name.clone(), value.clone());
    }

    // Read body
    let body_bytes = match axum::body::to_bytes(req.into_body(), MAX_BODY_SIZE).await {
        Ok(b) => b,
        Err(_) => {
            return json_error_response(StatusCode::BAD_REQUEST, "body_too_large");
        }
    };

    // Detect streaming from request body
    let is_streaming = detect_streaming(&body_bytes);

    // Build upstream URL
    let upstream_url = if let Some(q) = &query {
        format!("{}{}?{}", state.upstream_url, path, q)
    } else {
        format!("{}{}", state.upstream_url, path)
    };

    // 4. Forward to upstream
    let upstream_resp = match state
        .client
        .request(method, &upstream_url)
        .headers(upstream_headers)
        .body(body_bytes.to_vec())
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            let msg = format!("upstream_error: {}", e);
            return json_error_response(StatusCode::BAD_GATEWAY, &msg);
        }
    };

    // 5. If upstream error, pass through unchanged
    let status = upstream_resp.status();
    if status.is_client_error() || status.is_server_error() {
        return pass_through_response(upstream_resp).await;
    }

    // 6/7/8. Route to streaming or non-streaming handler
    if is_streaming {
        handle_streaming_response(state, upstream_resp).await
    } else {
        handle_non_streaming_response(state, upstream_resp).await
    }
}

/// Check if the request body contains `"stream": true`.
fn detect_streaming(body: &[u8]) -> bool {
    if let Ok(s) = std::str::from_utf8(body) {
        // Simple check: look for "stream":true or "stream": true in the JSON
        s.contains("\"stream\":true") || s.contains("\"stream\": true")
    } else {
        false
    }
}

/// Handle a non-streaming upstream response through the confinement pipeline.
async fn handle_non_streaming_response(
    state: Arc<ProxyState>,
    resp: reqwest::Response,
) -> Response<Body> {
    let status = resp.status();
    let headers = resp.headers().clone();

    let body_bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return json_error_response(
                StatusCode::BAD_GATEWAY,
                &format!("failed to read upstream body: {}", e),
            );
        }
    };

    // Try to parse as JSON
    let json_val: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => {
            // Not JSON — pass through unchanged
            return build_response_from_parts(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK),
                &headers,
                body_bytes.to_vec(),
            );
        }
    };

    // Extract text content
    let text = match extract_text_content(&json_val) {
        Some(t) => t,
        None => {
            // No text content — pass through unchanged
            return build_response_from_parts(
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK),
                &headers,
                body_bytes.to_vec(),
            );
        }
    };

    // Run through confinement pipeline
    let mut pipeline = state.pipeline.lock().await;
    match pipeline.confine(&text) {
        Ok(result) => {
            let state_str = result
                .state_elements
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let proof_hash = result.proof.as_ref().map(|p| p.trace_root.clone());

            let axum_status =
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK);
            let mut builder = Response::builder().status(axum_status);

            // Copy upstream headers
            for (name, value) in headers.iter() {
                builder = builder.header(name, value);
            }

            // Add confinement metadata headers
            builder = builder.header("x-gravrail-seq", result.seq.to_string());
            builder = builder.header("x-gravrail-state", format!("[{}]", state_str));
            if let Some(hash) = &proof_hash {
                builder = builder.header("x-gravrail-proof", hash.as_str());
            }

            builder
                .body(Body::from(body_bytes.to_vec()))
                .unwrap_or_else(|_| {
                    json_error_response(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "failed to build response",
                    )
                })
        }
        Err(ConfinementError::ReachabilityViolation { text_preview, .. }) => {
            json_error_response(
                StatusCode::FORBIDDEN,
                &format!("reachability_violation: {}", text_preview),
            )
        }
        Err(e) => json_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("confinement_error: {}", e),
        ),
    }
}

/// Handle a streaming (SSE) upstream response through the confinement pipeline.
async fn handle_streaming_response(
    state: Arc<ProxyState>,
    resp: reqwest::Response,
) -> Response<Body> {
    let mut upstream_stream = resp.bytes_stream();

    let stream = async_stream::stream! {
        let mut buffer = String::new();

        while let Some(chunk_result) = upstream_stream.next().await {
            // Check watchdog between chunks
            if !state.watchdog.is_alive() {
                let err_chunk = "data: {\"error\":\"proxy_killed\"}\n\n";
                yield Ok::<_, std::io::Error>(Bytes::from(err_chunk.to_string()));
                break;
            }

            let chunk_bytes = match chunk_result {
                Ok(b) => b,
                Err(e) => {
                    let err_chunk = format!("data: {{\"error\":\"{}\"}}\n\n", e);
                    yield Ok(Bytes::from(err_chunk));
                    break;
                }
            };

            let chunk_str = match std::str::from_utf8(&chunk_bytes) {
                Ok(s) => s.to_string(),
                Err(_) => {
                    // Binary chunk — pass through
                    yield Ok(Bytes::from(chunk_bytes.to_vec()));
                    continue;
                }
            };

            buffer.push_str(&chunk_str);

            // Process complete lines
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                // Try to parse as SSE data line
                match parse_sse_data(&line) {
                    Some(data_payload) => {
                        // Try to parse JSON and extract text
                        if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(&data_payload) {
                            if let Some(text) = extract_text_content(&json_val) {
                                // Confine the text
                                let mut pipeline = state.pipeline.lock().await;
                                match pipeline.confine(&text) {
                                    Ok(result) => {
                                        let state_elems: Vec<String> = result
                                            .state_elements
                                            .iter()
                                            .map(|e| e.to_string())
                                            .collect();
                                        let proof_hash = result
                                            .proof
                                            .as_ref()
                                            .map(|p| p.trace_root.as_str());
                                        let formatted = format_sse_chunk(
                                            &data_payload,
                                            result.seq,
                                            &state_elems,
                                            proof_hash,
                                        );
                                        yield Ok(Bytes::from(formatted));
                                    }
                                    Err(ConfinementError::ReachabilityViolation { text_preview, .. }) => {
                                        let err_chunk = format!(
                                            "data: {{\"error\":\"reachability_violation\",\"preview\":\"{}\"}}\n\n",
                                            text_preview.replace('"', "\\\"")
                                        );
                                        yield Ok(Bytes::from(err_chunk));
                                        return; // Kill the stream
                                    }
                                    Err(e) => {
                                        let err_chunk = format!(
                                            "data: {{\"error\":\"confinement_error\",\"detail\":\"{}\"}}\n\n",
                                            format!("{}", e).replace('"', "\\\"")
                                        );
                                        yield Ok(Bytes::from(err_chunk));
                                        return;
                                    }
                                }
                            } else {
                                // JSON but no text content — pass through
                                let passthrough = format!("data: {}\n\n", data_payload);
                                yield Ok(Bytes::from(passthrough));
                            }
                        } else {
                            // Not JSON — pass through
                            let passthrough = format!("data: {}\n\n", data_payload);
                            yield Ok(Bytes::from(passthrough));
                        }
                    }
                    None => {
                        // Not a data line (comment, empty, [DONE]) — pass through
                        let passthrough = format!("{}\n", line);
                        yield Ok(Bytes::from(passthrough));
                    }
                }
            }
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| {
            json_error_response(StatusCode::INTERNAL_SERVER_ERROR, "failed to build stream response")
        })
}

/// Pass through an upstream response unchanged.
async fn pass_through_response(resp: reqwest::Response) -> Response<Body> {
    let status = resp.status();
    let headers = resp.headers().clone();
    let body_bytes = resp.bytes().await.unwrap_or_default();
    build_response_from_parts(
        StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK),
        &headers,
        body_bytes.to_vec(),
    )
}

/// Build an axum Response from status, headers, and body bytes.
fn build_response_from_parts(
    status: StatusCode,
    headers: &reqwest::header::HeaderMap,
    body: Vec<u8>,
) -> Response<Body> {
    let mut builder = Response::builder().status(status);
    for (name, value) in headers.iter() {
        builder = builder.header(name.as_str(), value.as_bytes());
    }
    builder.body(Body::from(body)).unwrap_or_else(|_| {
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from("internal error"))
            .unwrap()
    })
}

/// Create a JSON error response.
fn json_error_response(status: StatusCode, message: &str) -> Response<Body> {
    let body = serde_json::json!({"error": message}).to_string();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(r#"{"error":"internal"}"#))
                .unwrap()
        })
}
