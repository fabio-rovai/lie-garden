use crate::lie::group::{LieGroup, GroupElement};

/// Generate a public key from a secret (algebra coefficients).
pub fn generate_public(group: &LieGroup, secret: &[f64]) -> GroupElement {
    group.exp(secret)
}

/// Compute shared secret from own secret and other party's public key.
/// Both parties arrive at the same shared secret because element-wise
/// multiplication in the algebra is commutative: a_i * b_i = b_i * a_i.
///
/// Alice: compute_shared(group, a, pub_b) = exp(a ⊙ log(pub_b)) = exp(a ⊙ b)
/// Bob:   compute_shared(group, b, pub_a) = exp(b ⊙ log(pub_a)) = exp(b ⊙ a)
pub fn compute_shared(group: &LieGroup, my_secret: &[f64], their_public: &GroupElement) -> GroupElement {
    let their_log = group.log(their_public);
    let shared_coeffs: Vec<f64> = my_secret.iter().zip(their_log.iter())
        .map(|(a, b)| a * b)
        .collect();
    group.exp(&shared_coeffs)
}

/// Convenience: run full key exchange given both secrets (for testing).
/// Returns (pub_a, pub_b, shared_a, shared_b).
pub fn diffie_hellman(
    group: &LieGroup,
    secret_a: &[f64],
    secret_b: &[f64],
) -> (GroupElement, GroupElement, GroupElement, GroupElement) {
    let pub_a = generate_public(group, secret_a);
    let pub_b = generate_public(group, secret_b);
    let shared_a = compute_shared(group, secret_a, &pub_b);
    let shared_b = compute_shared(group, secret_b, &pub_a);
    (pub_a, pub_b, shared_a, shared_b)
}
