/// Unit tests for crypto primitive modules:
///   sign.rs     — Schnorr signatures on Lie groups
///   commit.rs   — Pedersen commitments
///   keyexchange.rs — Diffie-Hellman key exchange
///   zkproof.rs  — Zero-knowledge proofs of discrete-log knowledge
///
/// These tests use fixed, deterministic inputs and cover:
///   - Happy-path roundtrips (valid output verifies)
///   - Rejection of tampered/wrong inputs
///   - Multiple group types (SO(3), SE(3), GL(2))
///   - Edge cases (zero vector, single-element, large coefficients)

use gravrail::crypto::commit::{commit, verify_commit};
use gravrail::crypto::keyexchange::{compute_shared, diffie_hellman, generate_public};
use gravrail::crypto::sign::{sign, verify};
use gravrail::crypto::zkproof::{prove_knowledge, verify_knowledge};
use gravrail::lie::group::{GroupType, LieGroup};

// ---------------------------------------------------------------------------
// sign.rs — Schnorr signatures
// ---------------------------------------------------------------------------

#[test]
fn sign_roundtrip_so3() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.4, 0.1, -0.3];
    let public = g.exp(&secret);
    let msg = b"gravrail unit test";
    let sig = sign(&g, msg, &secret);
    assert!(
        verify(&g, msg, &public, &sig),
        "valid SO(3) signature must verify"
    );
}

#[test]
fn sign_roundtrip_se3() {
    // SE(3): n=4, algebra_dim = 6
    let g = LieGroup::new(GroupType::SE, 4);
    let secret = vec![0.1, -0.2, 0.3, 0.05, -0.05, 0.1];
    let public = g.exp(&secret);
    let msg = b"se3 sign test";
    let sig = sign(&g, msg, &secret);
    assert!(
        verify(&g, msg, &public, &sig),
        "valid SE(3) signature must verify"
    );
}

#[test]
fn sign_roundtrip_gl2() {
    // GL(2): n=2, algebra_dim = 4
    let g = LieGroup::new(GroupType::GL, 2);
    let secret = vec![0.1, 0.2, -0.1, 0.3];
    let public = g.exp(&secret);
    let msg = b"gl2 sign test";
    let sig = sign(&g, msg, &secret);
    assert!(
        verify(&g, msg, &public, &sig),
        "valid GL(2) signature must verify"
    );
}

#[test]
fn sign_tampered_message_fails() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.4, 0.1, -0.3];
    let public = g.exp(&secret);
    let sig = sign(&g, b"original message", &secret);
    assert!(
        !verify(&g, b"tampered message", &public, &sig),
        "signature over different message must not verify"
    );
}

#[test]
fn sign_wrong_public_key_fails() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.4, 0.1, -0.3];
    let sig = sign(&g, b"msg", &secret);
    let wrong_public = g.exp(&[0.1, 0.2, 0.3]);
    assert!(
        !verify(&g, b"msg", &wrong_public, &sig),
        "signature verified against wrong public key must fail"
    );
}

#[test]
fn sign_zero_coefficients() {
    // Zero secret is degenerate but must not panic
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.0, 0.0, 0.0];
    let public = g.exp(&secret);
    let sig = sign(&g, b"zero secret", &secret);
    assert!(
        verify(&g, b"zero secret", &public, &sig),
        "zero-secret signature must still roundtrip"
    );
}

#[test]
fn sign_different_secrets_produce_different_signatures() {
    let g = LieGroup::new(GroupType::SO, 3);
    let s1 = vec![0.3, 0.1, -0.2];
    let s2 = vec![0.3, 0.2, -0.2]; // one coefficient differs
    let msg = b"same message";
    let sig1 = sign(&g, msg, &s1);
    let sig2 = sign(&g, msg, &s2);
    // Signatures should differ (compare s vectors)
    let different = sig1.s.iter().zip(sig2.s.iter()).any(|(a, b)| (a - b).abs() > 1e-15);
    assert!(different, "signatures from different secrets must differ");
}

// ---------------------------------------------------------------------------
// commit.rs — Pedersen commitments
// ---------------------------------------------------------------------------

#[test]
fn commit_roundtrip_so3() {
    let g = LieGroup::new(GroupType::SO, 3);
    let x = vec![0.3, -0.2, 0.5];
    let r = vec![0.1, 0.4, -0.1];
    let c = commit(&g, &x, &r);
    assert!(
        verify_commit(&g, &c, &x, &r),
        "commitment must verify with correct value and randomness"
    );
}

#[test]
fn commit_roundtrip_se3() {
    let g = LieGroup::new(GroupType::SE, 4);
    let x = vec![0.1, -0.1, 0.2, 0.3, -0.3, 0.05];
    let r = vec![0.5, -0.5, 0.1, 0.2, -0.2, 0.4];
    let c = commit(&g, &x, &r);
    assert!(
        verify_commit(&g, &c, &x, &r),
        "SE(3) commitment must verify with correct opening"
    );
}

#[test]
fn commit_wrong_value_fails() {
    let g = LieGroup::new(GroupType::SO, 3);
    let x = vec![0.3, -0.2, 0.5];
    let r = vec![0.1, 0.4, -0.1];
    let c = commit(&g, &x, &r);
    let wrong_x = vec![0.3, -0.2, 0.6]; // last coefficient differs
    assert!(
        !verify_commit(&g, &c, &wrong_x, &r),
        "commitment must reject wrong value"
    );
}

#[test]
fn commit_wrong_randomness_fails() {
    let g = LieGroup::new(GroupType::SO, 3);
    let x = vec![0.3, -0.2, 0.5];
    let r = vec![0.1, 0.4, -0.1];
    let c = commit(&g, &x, &r);
    let wrong_r = vec![0.9, 0.9, 0.9];
    assert!(
        !verify_commit(&g, &c, &x, &wrong_r),
        "commitment must reject wrong randomness"
    );
}

#[test]
fn commit_different_randomness_produces_different_commitments() {
    let g = LieGroup::new(GroupType::SO, 3);
    let x = vec![0.3, -0.2, 0.5];
    let r1 = vec![0.1, 0.4, -0.1];
    let r2 = vec![0.2, 0.4, -0.1]; // first coefficient differs
    let c1 = commit(&g, &x, &r1);
    let c2 = commit(&g, &x, &r2);
    assert!(
        !c1.close_to(&c2, 1e-10),
        "different randomness must produce different commitments (hiding)"
    );
}

#[test]
fn commit_same_inputs_produce_same_commitment() {
    // Commitments are deterministic given the same (x, r)
    let g = LieGroup::new(GroupType::SO, 3);
    let x = vec![0.1, 0.2, 0.3];
    let r = vec![0.4, 0.5, 0.6];
    let c1 = commit(&g, &x, &r);
    let c2 = commit(&g, &x, &r);
    assert!(
        c1.close_to(&c2, 1e-15),
        "commitment is deterministic — same inputs must produce same commitment"
    );
}

// ---------------------------------------------------------------------------
// keyexchange.rs — Diffie-Hellman
// ---------------------------------------------------------------------------

#[test]
fn dh_shared_secret_matches_so3() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret_a = vec![0.3, -0.1, 0.2];
    let secret_b = vec![-0.2, 0.4, 0.1];
    let (_pub_a, _pub_b, shared_a, shared_b) = diffie_hellman(&g, &secret_a, &secret_b);
    assert!(
        shared_a.close_to(&shared_b, 1e-8),
        "Alice and Bob must derive identical shared secret on SO(3)"
    );
}

#[test]
fn dh_shared_secret_matches_se3() {
    let g = LieGroup::new(GroupType::SE, 4);
    let secret_a = vec![0.1, -0.2, 0.3, 0.05, -0.05, 0.1];
    let secret_b = vec![-0.1, 0.2, -0.1, 0.3, 0.1, -0.2];
    let (_pub_a, _pub_b, shared_a, shared_b) = diffie_hellman(&g, &secret_a, &secret_b);
    assert!(
        shared_a.close_to(&shared_b, 1e-8),
        "Alice and Bob must derive identical shared secret on SE(3)"
    );
}

#[test]
fn dh_shared_secret_matches_gl2() {
    let g = LieGroup::new(GroupType::GL, 2);
    let secret_a = vec![0.2, 0.1, -0.1, 0.3];
    let secret_b = vec![-0.1, 0.2, 0.3, -0.2];
    let (_pub_a, _pub_b, shared_a, shared_b) = diffie_hellman(&g, &secret_a, &secret_b);
    assert!(
        shared_a.close_to(&shared_b, 1e-8),
        "Alice and Bob must derive identical shared secret on GL(2)"
    );
}

#[test]
fn dh_different_partners_produce_different_secrets() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret_a = vec![0.3, -0.1, 0.2];
    let secret_b = vec![-0.2, 0.4, 0.1];
    let secret_c = vec![0.5, 0.3, -0.4]; // third party

    let pub_b = generate_public(&g, &secret_b);
    let pub_c = generate_public(&g, &secret_c);

    let shared_with_b = compute_shared(&g, &secret_a, &pub_b);
    let shared_with_c = compute_shared(&g, &secret_a, &pub_c);

    assert!(
        !shared_with_b.close_to(&shared_with_c, 1e-10),
        "shared secret with B must differ from shared secret with C"
    );
}

#[test]
fn dh_public_key_differs_from_secret_exp() {
    // generate_public is just exp(secret); verify it actually differs for non-trivial secrets
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.3, -0.1, 0.2];
    let public = generate_public(&g, &secret);
    let identity = g.identity();
    assert!(
        !public.close_to(&identity, 1e-10),
        "public key from non-zero secret must not be identity"
    );
}

// ---------------------------------------------------------------------------
// zkproof.rs — Zero-knowledge proofs
// ---------------------------------------------------------------------------

#[test]
fn zk_proof_valid_so3() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.3, -0.2, 0.5];
    let public = g.exp(&secret);
    let proof = prove_knowledge(&g, &secret, &public);
    assert!(
        verify_knowledge(&g, &public, &proof),
        "valid ZK proof must verify on SO(3)"
    );
}

#[test]
fn zk_proof_valid_se3() {
    let g = LieGroup::new(GroupType::SE, 4);
    let secret = vec![0.1, -0.2, 0.3, 0.05, -0.05, 0.1];
    let public = g.exp(&secret);
    let proof = prove_knowledge(&g, &secret, &public);
    assert!(
        verify_knowledge(&g, &public, &proof),
        "valid ZK proof must verify on SE(3)"
    );
}

#[test]
fn zk_proof_valid_gl2() {
    let g = LieGroup::new(GroupType::GL, 2);
    let secret = vec![0.1, 0.2, -0.1, 0.3];
    let public = g.exp(&secret);
    let proof = prove_knowledge(&g, &secret, &public);
    assert!(
        verify_knowledge(&g, &public, &proof),
        "valid ZK proof must verify on GL(2)"
    );
}

#[test]
fn zk_proof_wrong_public_key_fails() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.3, -0.2, 0.5];
    let public = g.exp(&secret);
    let proof = prove_knowledge(&g, &secret, &public);
    let wrong_public = g.exp(&[0.1, 0.1, 0.1]);
    assert!(
        !verify_knowledge(&g, &wrong_public, &proof),
        "proof must not verify against a different public key"
    );
}

#[test]
fn zk_proof_tampered_response_fails() {
    use gravrail::crypto::zkproof::ZkProof;

    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.3, -0.2, 0.5];
    let public = g.exp(&secret);
    let proof = prove_knowledge(&g, &secret, &public);

    // Flip the sign on the first response coefficient
    let mut bad_response = proof.response.clone();
    bad_response[0] += 1.0;
    let tampered = ZkProof {
        commitment: proof.commitment,
        response: bad_response,
    };
    assert!(
        !verify_knowledge(&g, &public, &tampered),
        "tampered response must not verify"
    );
}

#[test]
fn zk_proof_deterministic() {
    // prove_knowledge uses a deterministic PRF nonce — same inputs yield same proof
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.3, -0.2, 0.5];
    let public = g.exp(&secret);
    let proof1 = prove_knowledge(&g, &secret, &public);
    let proof2 = prove_knowledge(&g, &secret, &public);
    let responses_match = proof1
        .response
        .iter()
        .zip(proof2.response.iter())
        .all(|(a, b)| (a - b).abs() < 1e-15);
    assert!(responses_match, "prove_knowledge must be deterministic");
}
