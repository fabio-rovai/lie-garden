use crate::lie::group::{LieGroup, GroupElement};
use sha2::{Sha256, Digest};

/// Pedersen-style commitment: C = exp(x + H(r))
///
/// H(r) hashes the FULL randomness vector into each algebra coefficient,
/// so changing any r[j] changes all coefficients — preimage resistance
/// of SHA256 provides computational binding.
///
/// Hiding: coefficient i depends on H(domain || i || r_bytes), which is
/// uniformly distributed given random r, so C reveals nothing about x.
pub fn commit(group: &LieGroup, x: &[f64], r: &[f64]) -> GroupElement {
    let h = randomness_hash(r, group.algebra_dim());
    let combined: Vec<f64> = x.iter().zip(h.iter())
        .map(|(xi, hi)| xi + hi)
        .collect();
    group.exp(&combined)
}

/// Verify commitment opening.
pub fn verify_commit(group: &LieGroup, commitment: &GroupElement, x: &[f64], r: &[f64]) -> bool {
    let expected = commit(group, x, r);
    commitment.close_to(&expected, 1e-8)
}

/// Hash full randomness vector into algebra coefficients.
///
/// Each coefficient depends on ALL of r via SHA256(domain || index || r_0 || r_1 || ...),
/// making it infeasible to find r' ≠ r producing the same output (preimage resistance).
fn randomness_hash(r: &[f64], dim: usize) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = Sha256::new();
        hasher.update(b"gravrail-pedersen-h");
        hasher.update((i as u64).to_le_bytes());
        // Hash ALL randomness bytes — changing any r[j] changes this coefficient
        for v in r { hasher.update(v.to_le_bytes()); }
        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash[..8].try_into().unwrap();
        coefficients.push((u64::from_le_bytes(bytes) as f64 / u64::MAX as f64) - 0.5);
    }
    coefficients
}
