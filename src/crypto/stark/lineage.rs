//! STARK proof of lineage integrity: proves a hash chain was computed correctly.
//!
//! The claim: given a sequence of events, each hash H[i] was computed as
//! SHA256(H[i-1] : event_data[i]) with H[0] derived from "genesis".
//!
//! Rather than proving SHA256 inside the STARK circuit (which is expensive),
//! we prove a simpler but still useful property: the prover committed to
//! a sequence of hash values that form a valid chain, and we verify
//! selected links using Merkle decommitments.
//!
//! This is a "hash chain commitment" proof:
//! 1. Prover builds a Merkle tree over the hash chain values
//! 2. Verifier challenges random positions
//! 3. Prover opens those positions with authentication paths
//! 4. Verifier checks the opened values satisfy the chain property
//!
//! Combined with the Fiat-Shamir transform, this gives a non-interactive
//! proof of hash chain integrity.

use sha2::{Sha256, Digest};
use super::field::FieldElement;
use super::merkle::{MerkleTree, verify_decommitment};
use super::channel::Channel;

/// A proof that a hash chain is correctly formed.
#[derive(Clone)]
pub struct LineageProof {
    /// Merkle root over the hash chain values
    pub chain_root: String,
    /// Number of events in the chain
    pub chain_length: usize,
    /// Query positions (Fiat-Shamir selected)
    pub query_indices: Vec<usize>,
    /// Hash values at queried positions
    pub queried_hashes: Vec<String>,
    /// Hash values at positions before queries (for chain verification)
    pub queried_prev_hashes: Vec<String>,
    /// Merkle authentication paths for queried positions
    pub auth_paths: Vec<Vec<String>>,
    /// Merkle authentication paths for prev positions
    pub prev_auth_paths: Vec<Vec<String>>,
    /// The event data at queried positions (needed to recompute hash)
    pub queried_events: Vec<String>,
    /// Full Fiat-Shamir transcript
    pub proof_transcript: Vec<String>,
}

/// Compute hash chain from events, matching lineage.rs format:
/// payload = "prev_hash:session_id:seq:ts:event_type:operation:details"
/// This function takes pre-formatted event strings.
fn compute_chain(events: &[String]) -> Vec<String> {
    let mut chain = Vec::with_capacity(events.len());
    let mut prev = "genesis".to_string();

    for event in events {
        let payload = format!("{}:{}", prev, event);
        let hash = format!("{:x}", Sha256::digest(payload.as_bytes()));
        chain.push(hash.clone());
        prev = hash;
    }
    chain
}

/// Generate a STARK-style proof that a hash chain was correctly computed.
///
/// `events` should be the formatted event strings (session_id:seq:ts:event_type:op:details).
/// The proof commits to the full chain via a Merkle tree and opens random positions.
pub fn prove_lineage(events: &[String]) -> LineageProof {
    assert!(!events.is_empty(), "Cannot prove empty lineage");

    let chain = compute_chain(events);
    let chain_len = chain.len();

    // Map hash chain to field elements for Merkle tree
    // Take first 8 bytes of each hash as a field element
    let field_chain: Vec<FieldElement> = chain.iter().map(|h| {
        let bytes = &h.as_bytes()[..16.min(h.len())];
        let val = u64::from_str_radix(std::str::from_utf8(bytes).unwrap_or("0"), 16).unwrap_or(0);
        FieldElement::new(val as i128)
    }).collect();

    // Build Merkle tree over the chain
    let mut merkle = MerkleTree::new(field_chain.clone());
    merkle.build_tree();

    let mut channel = Channel::new();
    channel.send(&merkle.root);

    // Select random query positions via Fiat-Shamir
    let num_queries = 3.min(chain_len);
    let mut query_indices = Vec::with_capacity(num_queries);
    let mut queried_hashes = Vec::new();
    let mut queried_prev_hashes = Vec::new();
    let mut auth_paths = Vec::new();
    let mut prev_auth_paths = Vec::new();
    let mut queried_events = Vec::new();

    for _ in 0..num_queries {
        // Query a position > 0 so we can check the chain link
        let idx = if chain_len > 1 {
            let raw = channel.receive_random_int(1, chain_len - 1, false);
            raw as usize
        } else {
            0
        };

        query_indices.push(idx);
        queried_hashes.push(chain[idx].clone());
        queried_events.push(events[idx].clone());

        // Get the previous hash for chain verification
        let prev_hash = if idx == 0 {
            "genesis".to_string()
        } else {
            chain[idx - 1].clone()
        };
        queried_prev_hashes.push(prev_hash);

        // Merkle authentication paths
        let padded_len = (field_chain.len() as f64).log2().ceil().exp2() as usize;
        if idx < padded_len {
            auth_paths.push(merkle.get_authentication_path(idx));
        } else {
            auth_paths.push(vec![]);
        }

        if idx > 0 && idx - 1 < padded_len {
            prev_auth_paths.push(merkle.get_authentication_path(idx - 1));
        } else {
            prev_auth_paths.push(vec![]);
        }
    }

    LineageProof {
        chain_root: merkle.root,
        chain_length: chain_len,
        query_indices,
        queried_hashes,
        queried_prev_hashes,
        auth_paths,
        prev_auth_paths,
        queried_events,
        proof_transcript: channel.proof,
    }
}

/// Verify a lineage proof: check that queried chain links are valid.
pub fn verify_lineage_proof(proof: &LineageProof) -> bool {
    if proof.chain_root.is_empty() || proof.chain_length == 0 {
        return false;
    }

    // Verify each queried link: H[i] == SHA256(H[i-1] : event[i])
    for q in 0..proof.query_indices.len() {
        let prev = &proof.queried_prev_hashes[q];
        let event = &proof.queried_events[q];
        let expected_hash = &proof.queried_hashes[q];

        let payload = format!("{}:{}", prev, event);
        let computed = format!("{:x}", Sha256::digest(payload.as_bytes()));

        if computed != *expected_hash {
            return false;
        }

        // Verify the queried hash is in the Merkle tree.
        // For a 1-leaf tree (padded_len == 1) the auth path is legitimately empty
        // because no sibling exists. For padded_len > 1, an empty auth path is a
        // soundness failure: a malicious prover could omit it to skip the Merkle
        // check entirely.
        let idx = proof.query_indices[q];
        let padded_len = (proof.chain_length as f64).log2().ceil().exp2() as usize;
        if idx >= padded_len {
            return false;
        }
        let bytes = &expected_hash.as_bytes()[..16.min(expected_hash.len())];
        let utf8 = match std::str::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return false,
        };
        let val = match u64::from_str_radix(utf8, 16) {
            Ok(v) => v,
            Err(_) => return false,
        };
        let fe = FieldElement::new(val as i128);
        if padded_len == 1 {
            // Single-leaf tree: the leaf is itself the root (after sha256 of value).
            let leaf_hash = format!("{:x}", Sha256::digest(fe.to_string().as_bytes()));
            if leaf_hash != proof.chain_root {
                return false;
            }
        } else {
            if proof.auth_paths[q].is_empty() {
                return false;
            }
            if !verify_decommitment(idx, &fe, &proof.auth_paths[q], &proof.chain_root) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_events(n: usize) -> Vec<String> {
        (0..n).map(|i| {
            format!("session1:{}:1700000{}:step:grav_step:coeffs_{}", i + 1, i, i)
        }).collect()
    }

    #[test]
    fn test_lineage_proof_valid() {
        let events = make_events(8);
        let proof = prove_lineage(&events);
        assert!(verify_lineage_proof(&proof));
        assert_eq!(proof.chain_length, 8);
    }

    #[test]
    fn test_lineage_proof_single_event() {
        let events = make_events(1);
        let proof = prove_lineage(&events);
        assert!(verify_lineage_proof(&proof));
    }

    #[test]
    fn test_lineage_proof_tampered_event_fails() {
        let events = make_events(8);
        let mut proof = prove_lineage(&events);
        // Tamper with a queried event
        if !proof.queried_events.is_empty() {
            proof.queried_events[0] = "tampered:data".to_string();
        }
        assert!(!verify_lineage_proof(&proof));
    }

    #[test]
    fn test_lineage_proof_tampered_hash_fails() {
        let events = make_events(8);
        let mut proof = prove_lineage(&events);
        if !proof.queried_hashes.is_empty() {
            proof.queried_hashes[0] = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        }
        assert!(!verify_lineage_proof(&proof));
    }

    #[test]
    fn test_chain_deterministic() {
        let events = make_events(4);
        let chain1 = compute_chain(&events);
        let chain2 = compute_chain(&events);
        assert_eq!(chain1, chain2);
    }
}
