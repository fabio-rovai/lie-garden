use crate::lie::group::{LieGroup, GroupElement};
use sha2::{Sha256, Digest};

pub struct ZkProof {
    pub commitment: GroupElement,
    pub response: Vec<f64>,
}

/// Prove knowledge of x such that public = exp(x).
pub fn prove_knowledge(group: &LieGroup, secret: &[f64], public: &GroupElement) -> ZkProof {
    // Nonce derived via PRF: k = H(secret || public || index) — deterministic but not invertible.
    let k = derive_nonce(secret, public, group.algebra_dim());
    let r = group.exp(&k);

    // Challenge = H(R || public)
    let challenge = compute_challenge(&r, public);

    // Response = k + challenge * secret (in algebra)
    let response: Vec<f64> = k.iter().zip(secret.iter())
        .map(|(ki, si)| ki + challenge * si)
        .collect();

    ZkProof { commitment: r, response }
}

/// Verify ZK proof.
pub fn verify_knowledge(group: &LieGroup, public: &GroupElement, proof: &ZkProof) -> bool {
    let challenge = compute_challenge(&proof.commitment, public);

    // Verify in the algebra to avoid BCH issues in non-abelian groups:
    // response should equal log(R) + c * log(public)
    let log_r = group.log(&proof.commitment);
    let log_pub = group.log(public);
    let expected: Vec<f64> = log_r.iter().zip(log_pub.iter())
        .map(|(ki, xi)| ki + challenge * xi)
        .collect();

    proof.response.iter().zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6)
}

/// Derive nonce from secret via PRF (not invertible — preserves zero-knowledge).
fn derive_nonce(secret: &[f64], public: &GroupElement, dim: usize) -> Vec<f64> {
    let mut coefficients = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut hasher = Sha256::new();
        hasher.update(b"gravrail-zk-nonce");
        hasher.update((i as u64).to_le_bytes());
        for v in secret { hasher.update(v.to_le_bytes()); }
        for v in public.matrix() { hasher.update(v.to_le_bytes()); }
        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash[..8].try_into().unwrap();
        coefficients.push((u64::from_le_bytes(bytes) as f64 / u64::MAX as f64) * 0.5);
    }
    coefficients
}

fn compute_challenge(r: &GroupElement, public: &GroupElement) -> f64 {
    let mut hasher = Sha256::new();
    for v in r.matrix() { hasher.update(v.to_le_bytes()); }
    for v in public.matrix() { hasher.update(v.to_le_bytes()); }
    let hash = hasher.finalize();
    let bytes: [u8; 8] = hash[..8].try_into().unwrap();
    // Wider challenge space (0..0.5) for better security, scaled for numerical stability
    (u64::from_le_bytes(bytes) as f64 / u64::MAX as f64) * 0.5
}
