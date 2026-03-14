use crate::lie::group::{LieGroup, GroupElement};
use sha2::{Sha256, Digest};

pub struct Signature {
    pub r: GroupElement,
    pub s: Vec<f64>,
}

pub fn sign(group: &LieGroup, message: &[u8], secret: &[f64]) -> Signature {
    // Nonce: per-coefficient PRF keyed on (message, secret, index) — unique bytes for every dim.
    let k: Vec<f64> = (0..secret.len()).map(|i| {
        let mut hasher = Sha256::new();
        hasher.update(b"gravrail-schnorr-nonce");
        hasher.update((i as u64).to_le_bytes());
        hasher.update(message);
        for v in secret { hasher.update(v.to_le_bytes()); }
        let hash = hasher.finalize();
        let bytes: [u8; 8] = hash[..8].try_into().unwrap();
        (u64::from_le_bytes(bytes) as f64 / u64::MAX as f64) * 0.5
    }).collect();

    let r = group.exp(&k);

    let challenge = hash_challenge(&r, message);

    let s: Vec<f64> = k.iter().zip(secret.iter())
        .map(|(ki, si)| ki + challenge * si)
        .collect();

    Signature { r, s }
}

pub fn verify(group: &LieGroup, message: &[u8], public: &GroupElement, sig: &Signature) -> bool {
    let challenge = hash_challenge(&sig.r, message);

    // Verify in the algebra to avoid BCH issues in non-abelian groups:
    // s should equal log(R) + c * log(public)
    let log_r = group.log(&sig.r);
    let log_pub = group.log(public);
    let expected_s: Vec<f64> = log_r.iter().zip(log_pub.iter())
        .map(|(ki, xi)| ki + challenge * xi)
        .collect();

    sig.s.iter().zip(expected_s.iter())
        .all(|(a, b)| {
            let mag = a.abs().max(b.abs());
            (a - b).abs() < mag * 1e-8 + 1e-12
        })
}

fn hash_challenge(r: &GroupElement, message: &[u8]) -> f64 {
    let mut hasher = Sha256::new();
    for v in r.matrix() { hasher.update(v.to_le_bytes()); }
    hasher.update(message);
    let hash = hasher.finalize();
    let bytes: [u8; 8] = hash[..8].try_into().unwrap();
    // Wider challenge space (0..0.5) for better security, scaled for numerical stability
    (u64::from_le_bytes(bytes) as f64 / u64::MAX as f64) * 0.5
}
