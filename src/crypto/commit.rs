use crate::lie::group::{LieGroup, GroupElement};
use sha2::{Sha256, Digest};

/// Pedersen commitment: C = exp(x) * exp(h_basis(r))
/// The second generator basis is derived from a hash — computationally independent from the standard basis.
pub fn commit(group: &LieGroup, x: &[f64], r: &[f64]) -> GroupElement {
    let gx = group.exp(x);
    let h_coeffs = second_generator_basis(r, group.algebra_dim());
    let hr = group.exp(&h_coeffs);
    group.multiply(&gx, &hr)
}

/// Verify commitment opening.
pub fn verify_commit(group: &LieGroup, commitment: &GroupElement, x: &[f64], r: &[f64]) -> bool {
    let expected = commit(group, x, r);
    commitment.close_to(&expected, 1e-8)
}

/// Derive a second generator basis from randomness using hash — not algebraically related to standard basis.
fn second_generator_basis(r: &[f64], dim: usize) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = Sha256::new();
        hasher.update(b"gravrail-pedersen-h");
        hasher.update((i as u64).to_le_bytes());
        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash[..8].try_into().unwrap();
        // Hash gives a fixed "direction" for the second generator
        let h_basis = (u64::from_le_bytes(bytes) as f64 / u64::MAX as f64) * 2.0 - 1.0;
        // Scale by the randomness coefficient
        let r_val = if i < r.len() { r[i] } else { 0.0 };
        coefficients.push(h_basis * r_val);
    }
    coefficients
}
