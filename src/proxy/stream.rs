// SSE chunk buffering and confined forwarding

use serde_json::Value;

/// Extract text content from an OpenAI-format JSON response.
/// Handles streaming (`choices[0].delta.content`) and
/// non-streaming (`choices[0].message.content`) variants.
/// Returns None for empty content or missing fields.
pub fn extract_text_content(json: &Value) -> Option<String> {
    let choice = json.get("choices")?.get(0)?;

    // Try streaming format first (delta.content), then non-streaming (message.content)
    let content = choice
        .get("delta")
        .and_then(|d| d.get("content"))
        .or_else(|| choice.get("message").and_then(|m| m.get("content")))?;

    let s = content.as_str()?;
    if s.is_empty() {
        return None;
    }
    Some(s.to_string())
}

/// Parse an SSE line and return the data payload if present.
/// Returns None for `data: [DONE]`, comments (`: ...`), and empty lines.
pub fn parse_sse_data(line: &str) -> Option<String> {
    if line.is_empty() || line.starts_with(": ") || line == ":" {
        return None;
    }
    let payload = line.strip_prefix("data: ")?;
    if payload == "[DONE]" {
        return None;
    }
    Some(payload.to_string())
}

/// Format an SSE chunk with confinement metadata prepended as SSE comments.
pub fn format_sse_chunk(
    original_data: &str,
    seq: u64,
    state_elements: &[String],
    proof_hash: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(": x-gravrail-seq={}\n", seq));
    out.push_str(&format!(": x-gravrail-state=[{}]\n", state_elements.join(",")));
    if let Some(hash) = proof_hash {
        out.push_str(&format!(": x-gravrail-proof={}\n", hash));
    }
    out.push_str(&format!("data: {}\n\n", original_data));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_extract_streaming_content() {
        let j = json!({"choices": [{"delta": {"content": "Hello"}}]});
        assert_eq!(extract_text_content(&j), Some("Hello".to_string()));
    }

    #[test]
    fn test_extract_non_streaming_content() {
        let j = json!({"choices": [{"message": {"content": "world"}}]});
        assert_eq!(extract_text_content(&j), Some("world".to_string()));
    }

    #[test]
    fn test_extract_empty_delta() {
        let j = json!({"choices": [{"delta": {"content": ""}}]});
        assert_eq!(extract_text_content(&j), None);
    }

    #[test]
    fn test_extract_no_content_field() {
        let j = json!({"choices": [{"delta": {"role": "assistant"}}]});
        assert_eq!(extract_text_content(&j), None);
    }

    #[test]
    fn test_parse_sse_data_line() {
        let line = r#"data: {"id":"chatcmpl-123"}"#;
        assert_eq!(parse_sse_data(line), Some(r#"{"id":"chatcmpl-123"}"#.to_string()));
    }

    #[test]
    fn test_parse_sse_done() {
        assert_eq!(parse_sse_data("data: [DONE]"), None);
    }

    #[test]
    fn test_parse_sse_comment() {
        assert_eq!(parse_sse_data(": comment"), None);
    }

    #[test]
    fn test_parse_sse_empty() {
        assert_eq!(parse_sse_data(""), None);
    }

    #[test]
    fn test_format_sse_chunk() {
        let result = format_sse_chunk(
            r#"{"text":"hi"}"#,
            42,
            &["e1".to_string(), "e2".to_string()],
            Some("abc123"),
        );
        assert!(result.contains(": x-gravrail-seq=42\n"));
        assert!(result.contains(": x-gravrail-state=[e1,e2]\n"));
        assert!(result.contains(": x-gravrail-proof=abc123\n"));
        assert!(result.contains("data: {\"text\":\"hi\"}\n\n"));
    }

    #[test]
    fn test_format_sse_chunk_no_proof() {
        let result = format_sse_chunk(
            r#"{"text":"hi"}"#,
            1,
            &["e1".to_string()],
            None,
        );
        assert!(result.contains(": x-gravrail-seq=1\n"));
        assert!(result.contains(": x-gravrail-state=[e1]\n"));
        assert!(!result.contains("x-gravrail-proof"));
        assert!(result.contains("data: {\"text\":\"hi\"}\n\n"));
    }
}
