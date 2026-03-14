use gravrail::crypto::commit::{commit, verify_commit};
use gravrail::crypto::keyexchange::diffie_hellman;
use gravrail::crypto::zkproof::{prove_knowledge, verify_knowledge};
use gravrail::crypto::sign::{sign, verify};
use gravrail::lie::group::{LieGroup, GroupType};
use proptest::prelude::*;

proptest! {
    #[test]
    fn commitment_binding(
        x in prop::collection::vec(-1.0f64..1.0, 3),
        r in prop::collection::vec(-1.0f64..1.0, 3),
        x2 in prop::collection::vec(-1.0f64..1.0, 3),
    ) {
        let g = LieGroup::new(GroupType::SO, 3);
        let c = commit(&g, &x, &r);
        prop_assert!(verify_commit(&g, &c, &x, &r));
        if x != x2 {
            prop_assert!(!verify_commit(&g, &c, &x2, &r));
        }
    }

    #[test]
    fn dh_key_agreement(
        a in prop::collection::vec(-1.0f64..1.0, 3),
        b in prop::collection::vec(-1.0f64..1.0, 3),
    ) {
        let g = LieGroup::new(GroupType::SO, 3);
        let (_pub_a, _pub_b, shared_a, shared_b) = diffie_hellman(&g, &a, &b);
        for (x, y) in shared_a.matrix().iter().zip(shared_b.matrix().iter()) {
            prop_assert!((x - y).abs() < 1e-8, "DH shared secret must agree");
        }
    }
}

#[test]
fn zk_proof_valid() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.3, -0.2, 0.5];
    let public = g.exp(&secret);
    let proof = prove_knowledge(&g, &secret, &public);
    assert!(verify_knowledge(&g, &public, &proof));
}

#[test]
fn schnorr_signature_valid() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.4, 0.1, -0.3];
    let public = g.exp(&secret);
    let msg = b"hello gravrail";
    let sig = sign(&g, msg, &secret);
    assert!(verify(&g, msg, &public, &sig));
}

#[test]
fn schnorr_signature_wrong_message() {
    let g = LieGroup::new(GroupType::SO, 3);
    let secret = vec![0.4, 0.1, -0.3];
    let public = g.exp(&secret);
    let sig = sign(&g, b"hello", &secret);
    assert!(!verify(&g, b"world", &public, &sig));
}
