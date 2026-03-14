// Session token generation and validation

use rand::Rng;

/// Session-scoped authentication token.
///
/// A random 32-hex-character token is generated at startup and printed to
/// stderr. The consumer must send it as `X-GravRail-Token` on the first
/// request. After the first successful validation the session is considered
/// authenticated and all subsequent calls pass without re-checking.
pub struct SessionAuth {
    token: String,
    validated: bool,
}

impl SessionAuth {
    /// Generate a new session token from OS randomness.
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let bytes: [u8; 16] = rng.r#gen();
        let token = hex::encode(bytes);
        Self {
            token,
            validated: false,
        }
    }

    /// Return the session token string (32 hex chars).
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Validate a candidate token using constant-time comparison.
    ///
    /// After the first successful match, all subsequent calls return `true`
    /// regardless of the candidate value.
    pub fn validate(&mut self, candidate: &str) -> bool {
        if self.validated {
            return true;
        }
        if constant_time_eq(candidate.as_bytes(), self.token.as_bytes()) {
            self.validated = true;
            true
        } else {
            false
        }
    }

    /// Whether this session has already been authenticated.
    pub fn is_validated(&self) -> bool {
        self.validated
    }
}

/// Timing-safe byte-level equality check.
///
/// Returns `false` immediately if lengths differ (this leaks length info,
/// which is acceptable here since token length is public knowledge).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_32_hex_chars() {
        let auth = SessionAuth::new();
        let token = auth.token();
        assert_eq!(token.len(), 32);
        assert!(token.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_correct_token_validates() {
        let mut auth = SessionAuth::new();
        let token = auth.token().to_owned();
        assert!(auth.validate(&token));
        assert!(auth.is_validated());
    }

    #[test]
    fn test_wrong_token_rejected() {
        let mut auth = SessionAuth::new();
        assert!(!auth.validate("00000000000000000000000000000000"));
        assert!(!auth.is_validated());
    }

    #[test]
    fn test_after_validation_all_pass() {
        let mut auth = SessionAuth::new();
        let token = auth.token().to_owned();
        assert!(auth.validate(&token));
        // After validation, even a wrong token should pass
        assert!(auth.validate("totally_wrong"));
        assert!(auth.validate(""));
    }

    #[test]
    fn test_constant_time_eq() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(constant_time_eq(b"", b""));
    }
}
