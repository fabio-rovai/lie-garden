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
use super::confine::{ConfinementPipeline, ConfinementError, InputResult};
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
    /// Output-side holonomy threshold. 0.0 = disabled.
    pub holonomy_threshold: f64,
    /// Input-side holonomy threshold. 0.0 = disabled.
    pub input_holonomy_threshold: f64,
    /// Input-side norm threshold. 0.0 = disabled.
    pub input_norm_threshold: f64,
    pub db: Option<Arc<crate::state::StateDb>>,
    pub session_id: String,
    pub circuit_id: String,
}

/// Build the Axum router with a catch-all handler.
pub fn build_router(state: Arc<ProxyState>) -> Router {
    Router::new()
        .route("/{*path}", any(proxy_handler))
        .with_state(state)
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_SIZE))
}

/// Parse an OpenAI-compatible JSON body and return the content of the last
/// message with `"role": "user"`, or `None` if not found.
pub(crate) fn extract_last_user_message(body: &[u8]) -> Option<String> {
    let json_val: serde_json::Value = serde_json::from_slice(body).ok()?;
    let messages = json_val.get("messages")?.as_array()?;
    messages
        .iter()
        .rev()
        .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))
        .and_then(|msg| msg.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
}

/// Compute cosine distance (1.0 - cosine_similarity) between two vectors.
///
/// Returns 0.0 if either vector is zero-length or the slice lengths differ.
pub(crate) fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    let similarity = dot / (norm_a * norm_b);
    1.0 - similarity.clamp(-1.0, 1.0)
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

    // 3b. Process user input through the input-side pipeline
    let input_result: Option<InputResult> =
        if let Some(user_text) = extract_last_user_message(&body_bytes) {
            let result = {
                let mut pipeline = state.pipeline.lock().await;
                pipeline.process_input(&user_text)
            };

            // Check input holonomy threshold
            if state.input_holonomy_threshold > 0.0
                && result.holonomy > state.input_holonomy_threshold
            {
                return block_response("input-holonomy");
            }

            // Check input norm threshold
            if state.input_norm_threshold > 0.0 && result.norm > state.input_norm_threshold {
                return block_response("input-norm");
            }

            Some(result)
        } else {
            None
        };

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
        handle_streaming_response(state, upstream_resp, input_result).await
    } else {
        handle_non_streaming_response(state, upstream_resp, input_result).await
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

/// Build a 429 blocking response with the given X-GravRail-Block header value.
fn block_response(reason: &str) -> Response<Body> {
    let body = serde_json::json!({"error": format!("blocked: {}", reason)}).to_string();
    Response::builder()
        .status(StatusCode::TOO_MANY_REQUESTS)
        .header("content-type", "application/json")
        .header("x-gravrail-block", reason)
        .body(Body::from(body))
        .unwrap_or_else(|_| {
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(r#"{"error":"internal"}"#))
                .unwrap()
        })
}

/// Handle a non-streaming upstream response through the confinement pipeline.
async fn handle_non_streaming_response(
    state: Arc<ProxyState>,
    resp: reqwest::Response,
    input_result: Option<InputResult>,
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
            // NOTE: confine() has already advanced the group state and seq when this
            // check fires. The state mutation is intentional — the block is an
            // observation-level gate, not a rollback. Task 6 (trajectory persistence)
            // should record blocked steps with a "blocked" flag so replay is consistent.
            // Check output holonomy threshold
            if state.holonomy_threshold > 0.0
                && result.output_holonomy > state.holonomy_threshold
            {
                drop(pipeline);
                return block_response("output-holonomy");
            }
            drop(pipeline);

            let state_str = result
                .state_elements
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let proof_hash = result.proof.as_ref().map(|p| p.trace_root.clone());

            // Compute drift between input and output coefficient vectors
            let drift_val: Option<f64> = if let Some(ref ir) = input_result {
                Some(cosine_distance(&ir.coeffs, &result.raw_coeffs))
            } else {
                None
            };
            // Keep legacy drift for headers (0.0 if no input)
            let drift = drift_val.unwrap_or(0.0);

            // Persist trajectory step (fire-and-forget)
            if let Some(db) = &state.db {
                let db = Arc::clone(db);
                let session_id = state.session_id.clone();
                let circuit_id = state.circuit_id.clone();
                let step = result.seq;
                let i_norm = input_result.as_ref().map(|ir| ir.norm);
                let i_hol = input_result.as_ref().map(|ir| ir.holonomy);
                let o_norm: f64 = result.state_elements.iter().map(|x| x * x).sum::<f64>().sqrt();
                let o_hol = Some(result.output_holonomy);
                let raw = result.raw_coeffs.clone();
                tokio::task::spawn_blocking(move || {
                    if let Err(e) = db.insert_trajectory_step(
                        &session_id, &circuit_id, step,
                        i_norm, i_hol, o_norm, o_hol, drift_val,
                        &raw,
                    ) {
                        eprintln!("[gravrail-proxy] trajectory persist error: {}", e);
                    }
                });
            }

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

            // Add holonomy / norm / drift headers
            builder = builder.header("x-gravrail-holonomy", result.output_holonomy.to_string());
            if let Some(ref ir) = input_result {
                builder = builder.header("x-gravrail-input-norm", ir.norm.to_string());
                builder = builder.header("x-gravrail-input-holonomy", ir.holonomy.to_string());
            }
            builder = builder.header("x-gravrail-drift", drift.to_string());

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
    // NOTE: _input_result is intentionally unused. process_input() was already
    // called and the input_window was updated, but SSE responses do not emit
    // per-chunk input holonomy headers. This means the input window advances
    // even for streaming requests. Future work: add holonomy to SSE metadata.
    _input_result: Option<InputResult>,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_last_user_message ────────────────────────────────────────────

    #[test]
    fn test_extract_last_user_message_basic() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"}
            ]
        })
        .to_string();
        let result = extract_last_user_message(body.as_bytes());
        assert_eq!(result, Some("Hello!".to_string()));
    }

    #[test]
    fn test_extract_last_user_message_picks_last() {
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second message"}
            ]
        })
        .to_string();
        let result = extract_last_user_message(body.as_bytes());
        assert_eq!(result, Some("Second message".to_string()));
    }

    #[test]
    fn test_extract_last_user_message_no_user_role() {
        let body = serde_json::json!({
            "messages": [
                {"role": "system", "content": "System only"},
                {"role": "assistant", "content": "Assistant only"}
            ]
        })
        .to_string();
        let result = extract_last_user_message(body.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_last_user_message_no_messages_field() {
        let body = serde_json::json!({"model": "gpt-4", "prompt": "hello"}).to_string();
        let result = extract_last_user_message(body.as_bytes());
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_last_user_message_invalid_json() {
        let result = extract_last_user_message(b"not json at all");
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_last_user_message_empty_body() {
        let result = extract_last_user_message(b"");
        assert!(result.is_none());
    }

    // ── cosine_distance ──────────────────────────────────────────────────────

    #[test]
    fn test_cosine_distance_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        // Identical vectors → cosine similarity = 1 → distance = 0
        let d = cosine_distance(&v, &v);
        assert!((d - 0.0).abs() < 1e-12, "identical vectors should give distance 0, got {}", d);
    }

    #[test]
    fn test_cosine_distance_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        // Opposite vectors → similarity = -1 → distance = 2
        let d = cosine_distance(&a, &b);
        assert!((d - 2.0).abs() < 1e-12, "opposite vectors should give distance 2, got {}", d);
    }

    #[test]
    fn test_cosine_distance_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // Orthogonal → similarity = 0 → distance = 1
        let d = cosine_distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-12, "orthogonal vectors should give distance 1, got {}", d);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = cosine_distance(&a, &b);
        assert_eq!(d, 0.0, "zero vector should return 0");
    }

    #[test]
    fn test_cosine_distance_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = cosine_distance(&a, &b);
        assert_eq!(d, 0.0, "mismatched lengths should return 0");
    }

    #[test]
    fn test_cosine_distance_empty_slices() {
        let d = cosine_distance(&[], &[]);
        assert_eq!(d, 0.0, "empty slices should return 0");
    }

    #[test]
    fn test_cosine_distance_symmetric() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d_ab = cosine_distance(&a, &b);
        let d_ba = cosine_distance(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-12, "cosine distance should be symmetric");
    }
}
