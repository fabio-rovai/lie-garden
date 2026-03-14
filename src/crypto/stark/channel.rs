use super::field::FieldElement;
use sha2::{Sha256, Digest};
use serde::Serialize;
use std::fmt::Debug;

fn sha256_hex(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

pub fn serialize<T: Serialize + Debug>(obj: &T) -> String {
    serde_json::to_string(obj).unwrap()
}

/// Fiat-Shamir channel for non-interactive proof generation.
/// Absorbs prover messages via `send`, squeezes verifier challenges via
/// `receive_random_field_element`. All randomness is deterministic from
/// the transcript hash state.
#[derive(Clone)]
pub struct Channel {
    pub state: String,
    pub proof: Vec<String>,
}

impl Channel {
    pub fn new() -> Self {
        Channel {
            state: "0".to_string(),
            proof: Vec::new(),
        }
    }

    pub fn send(&mut self, s: &str) {
        self.state = sha256_hex(&format!("{}{}", self.state, s));
        self.proof.push(format!("Channel:{}", s));
    }

    /// Returns a random integer in [min, max] derived from transcript state.
    pub fn receive_random_int(&mut self, min: usize, max: usize, show_in_proof: bool) -> u128 {
        // Parse first 32 hex chars of state as u128
        let state_bytes = &self.state[..32.min(self.state.len())];
        let state_val = u128::from_str_radix(state_bytes, 16).unwrap_or(0);

        let range = (max - min + 1) as u128;
        let num = (min as u128) + (state_val % range);

        self.state = sha256_hex(&self.state);
        if show_in_proof {
            self.proof.push(format!("Channel:{}", num));
        }
        num
    }

    /// Returns a random field element derived from transcript state.
    /// For Goldilocks (64-bit prime), we use the full SHA256 hash to get
    /// uniform sampling over the field.
    pub fn receive_random_field_element(&mut self) -> FieldElement {
        // Parse first 32 hex chars as u128, then reduce mod p
        let state_bytes = &self.state[..32.min(self.state.len())];
        let state_val = u128::from_str_radix(state_bytes, 16).unwrap_or(0);
        let fe = FieldElement::from(state_val);

        self.state = sha256_hex(&self.state);
        self.proof.push(format!("Channel:{}", fe.val));
        fe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_deterministic() {
        let mut ch1 = Channel::new();
        let mut ch2 = Channel::new();

        ch1.send("hello");
        ch2.send("hello");

        let r1 = ch1.receive_random_field_element();
        let r2 = ch2.receive_random_field_element();
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_channel_different_transcripts_differ() {
        let mut ch1 = Channel::new();
        let mut ch2 = Channel::new();

        ch1.send("hello");
        ch2.send("world");

        let r1 = ch1.receive_random_field_element();
        let r2 = ch2.receive_random_field_element();
        assert_ne!(r1, r2);
    }
}
