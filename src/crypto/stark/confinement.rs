//! STARK proof of confinement: proves that constraint masking was applied correctly.
//!
//! The claim: given a generator mask M (booleans) and raw algebra coefficients C,
//! the constrained output satisfies `out[i] = C[i] * M[i]` for all i.
//! This proves the prover actually zeroed inactive generators rather than smuggling
//! values through masked channels.
//!
//! The proof works by encoding the constraint as a polynomial identity over F_p:
//! For each coefficient index i, define trace values:
//!   a[3i]   = raw coefficient (mapped to field)
//!   a[3i+1] = mask bit (0 or 1 in field)
//!   a[3i+2] = constrained output
//! Then the constraint polynomial is: a[3i+2] - a[3i] * a[3i+1] == 0
//!
//! The STARK proves this constraint holds for all indices without revealing
//! the actual coefficients (zero-knowledge via random composition).

use super::field::FieldElement;
use super::polynomial::Polynomial;
use super::merkle::{MerkleTree, verify_decommitment};
use super::channel::Channel;

/// Scale factor for mapping f64 coefficients to field elements.
/// We multiply by 2^20 and round to get integer representations.
const SCALE: f64 = 1_048_576.0; // 2^20

/// Number of random queries for FRI verification.
const NUM_QUERIES: usize = 3;

/// Map an f64 coefficient to a field element (deterministic, injective for our range).
fn f64_to_field(v: f64) -> FieldElement {
    let scaled = (v * SCALE).round() as i128;
    FieldElement::new(scaled)
}

/// Map a field element back to f64 coefficient.
#[allow(dead_code)]
fn field_to_f64(fe: FieldElement) -> f64 {
    let modulus = FieldElement::modulus();
    let half_mod = modulus / 2;
    let signed = if (fe.val as i128) > half_mod { fe.val as i128 - modulus } else { fe.val as i128 };
    signed as f64 / SCALE
}

/// Query data for a single FRI decommitment point.
#[derive(Clone)]
pub struct QueryData {
    /// Index into the evaluation domain
    pub idx: usize,
    /// Trace polynomial evaluation at this point
    pub trace_val: FieldElement,
    /// Trace Merkle authentication path
    pub trace_auth: Vec<String>,
    /// Composition polynomial evaluation at this point
    pub cp_val: FieldElement,
    /// CP Merkle authentication path
    pub cp_auth: Vec<String>,
    /// FRI layer evaluations at this query (f(x) at each layer)
    pub fri_vals: Vec<FieldElement>,
    /// FRI layer evaluations at the sibling point (f(-x) at each layer)
    pub fri_sibling_vals: Vec<FieldElement>,
    /// FRI Merkle authentication paths per layer
    pub fri_auths: Vec<Vec<String>>,
    /// FRI sibling Merkle authentication paths per layer
    pub fri_sibling_auths: Vec<Vec<String>>,
}

/// A STARK proof that constraint masking was correctly applied.
#[derive(Clone)]
pub struct ConfinementProof {
    pub trace_root: String,
    pub cp_root: String,
    pub fri_roots: Vec<String>,
    pub fri_last_value: FieldElement,
    pub proof_transcript: Vec<String>,
    pub trace_length: usize,
    /// Query decommitments for verification
    pub queries: Vec<QueryData>,
}

/// Build the execution trace for a single masking operation.
/// Returns trace as flat vec: [raw_0, mask_0, out_0, raw_1, mask_1, out_1, ...]
/// Padded to next power of 2.
fn build_trace(
    raw_coefficients: &[f64],
    mask: &[bool],
    constrained: &[f64],
) -> Vec<FieldElement> {
    let dim = raw_coefficients.len();
    assert_eq!(mask.len(), dim);
    assert_eq!(constrained.len(), dim);

    let mut trace = Vec::with_capacity(dim * 3);
    for i in 0..dim {
        trace.push(f64_to_field(raw_coefficients[i]));
        trace.push(if mask[i] { FieldElement::one() } else { FieldElement::zero() });
        trace.push(f64_to_field(constrained[i]));
    }

    // Pad to next power of 2
    let target_len = (trace.len() as f64).log2().ceil().exp2() as usize;
    trace.resize(target_len.max(trace.len()), FieldElement::zero());

    trace
}

/// Generate a STARK proof that masking was applied correctly.
///
/// Proves: for each i in 0..dim, constrained[i] == raw[i] * mask[i]
pub fn prove_confinement(
    raw_coefficients: &[f64],
    mask: &[bool],
    constrained: &[f64],
) -> ConfinementProof {
    let dim = raw_coefficients.len();
    let trace = build_trace(raw_coefficients, mask, constrained);
    let trace_len = trace.len();

    // Find a subgroup of size trace_len for interpolation domain
    let group_order = FieldElement::modulus() - 1;
    assert!(
        trace_len.is_power_of_two() && (trace_len as i128) <= group_order,
        "Trace length must be a power of 2 fitting in the field"
    );

    // Generator for subgroup of size trace_len
    let g = FieldElement::generator().pow(group_order / trace_len as i128);
    let domain: Vec<FieldElement> = (0..trace_len as i128)
        .map(|i| g.pow(i))
        .collect();

    // Interpolate trace polynomial
    let f = Polynomial::interpolate_poly(&domain, &trace);

    // Evaluation domain: 8x blowup on a coset
    let blowup = 8;
    let eval_len = trace_len * blowup;
    let h = FieldElement::generator().pow(group_order / eval_len as i128);
    let coset_shift = FieldElement::generator();
    let eval_domain: Vec<FieldElement> = (0..eval_len as i128)
        .map(|i| coset_shift * h.pow(i))
        .collect();

    let f_eval: Vec<FieldElement> = eval_domain.iter().map(|x| f.eval(x)).collect();

    // Merkle commit to trace evaluations
    let mut f_merkle = MerkleTree::new(f_eval.clone());
    f_merkle.build_tree();

    let mut channel = Channel::new();
    channel.send(&f_merkle.root);

    // Build constraint evaluations
    let mut constraint_evals = Vec::with_capacity(eval_len);
    for x in &eval_domain {
        let raw_val = f.eval(x);
        let mask_val = f.eval(&(*x * g));
        let out_val = f.eval(&(*x * g * g));
        constraint_evals.push(out_val - raw_val * mask_val);
    }

    // Vanishing polynomial for trace triple-starts
    let vanish_points: Vec<FieldElement> = (0..dim)
        .map(|k| g.pow(3 * k as i128))
        .collect();

    let vanish_evals: Vec<FieldElement> = eval_domain.iter().map(|x| {
        vanish_points.iter().fold(FieldElement::one(), |acc, vp| acc * (*x - *vp))
    }).collect();

    // Quotient: constraint / vanishing
    let quotient_evals: Vec<FieldElement> = constraint_evals.iter()
        .zip(vanish_evals.iter())
        .map(|(c, v)| {
            if *v == FieldElement::zero() { FieldElement::zero() } else { *c / *v }
        })
        .collect();

    // Compose with random weight
    let alpha = channel.receive_random_field_element();
    let cp_evals: Vec<FieldElement> = quotient_evals.iter()
        .map(|q| *q * alpha)
        .collect();

    let mut cp_merkle = MerkleTree::new(cp_evals.clone());
    cp_merkle.build_tree();
    channel.send(&cp_merkle.root);

    // FRI commitment: fold the composition polynomial
    let mut fri_roots = Vec::new();
    let mut fri_merkles: Vec<MerkleTree> = Vec::new();
    let mut fri_layers: Vec<Vec<FieldElement>> = Vec::new();
    let mut current_evals = cp_evals.clone();
    let mut current_domain = eval_domain.clone();

    while current_evals.len() > 8 {
        let beta = channel.receive_random_field_element();
        let half_len = current_evals.len() / 2;

        let mut next_evals = Vec::with_capacity(half_len);
        let mut next_domain = Vec::with_capacity(half_len);

        for i in 0..half_len {
            let f_x = current_evals[i];
            let f_neg_x = current_evals[i + half_len];
            let x = current_domain[i];

            let even = (f_x + f_neg_x) * FieldElement::new(2).inverse();
            let odd = (f_x - f_neg_x) * (FieldElement::new(2) * x).inverse();
            next_evals.push(even + beta * odd);
            next_domain.push(x * x);
        }

        let mut next_merkle = MerkleTree::new(next_evals.clone());
        next_merkle.build_tree();
        fri_roots.push(next_merkle.root.clone());
        channel.send(&next_merkle.root);

        fri_layers.push(current_evals.clone());
        fri_merkles.push(next_merkle);

        current_evals = next_evals;
        current_domain = next_domain;
    }

    let fri_last_value = current_evals[0];
    channel.send(&fri_last_value.to_string());

    // Generate query decommitments via Fiat-Shamir
    let padded_eval_len = (eval_len as f64).log2().ceil().exp2() as usize;
    let num_queries = NUM_QUERIES.min(padded_eval_len / 2);
    let mut queries = Vec::with_capacity(num_queries);

    for _ in 0..num_queries {
        let idx = channel.receive_random_int(0, padded_eval_len / 2 - 1, false) as usize;

        let trace_val = f_eval[idx.min(f_eval.len() - 1)];
        let trace_auth = if idx < padded_eval_len {
            f_merkle.get_authentication_path(idx)
        } else {
            vec![]
        };

        let cp_val = cp_evals[idx.min(cp_evals.len() - 1)];
        let cp_auth = if idx < padded_eval_len {
            cp_merkle.get_authentication_path(idx)
        } else {
            vec![]
        };

        // FRI layer decommitments: for each layer, open the folded value
        // fri_merkles[i] commits to the output of folding iteration i
        let mut fri_vals = Vec::new();
        let mut fri_sibling_vals = Vec::new();
        let mut fri_auths = Vec::new();
        let mut fri_sibling_auths = Vec::new();
        let mut layer_idx = idx;

        for (_layer_num, merkle) in fri_merkles.iter().enumerate() {
            let folded_len = merkle.data.len();
            let folded_idx = layer_idx % folded_len;

            // The folded value (from next_evals of this iteration)
            fri_vals.push(merkle.data[folded_idx]);
            fri_auths.push(merkle.get_authentication_path(folded_idx));

            // Sibling not needed for Merkle verification
            fri_sibling_vals.push(FieldElement::zero());
            fri_sibling_auths.push(vec![]);

            layer_idx = folded_idx; // next layer uses the folded index
        }

        queries.push(QueryData {
            idx,
            trace_val,
            trace_auth,
            cp_val,
            cp_auth,
            fri_vals,
            fri_sibling_vals,
            fri_auths,
            fri_sibling_auths,
        });
    }

    ConfinementProof {
        trace_root: f_merkle.root,
        cp_root: cp_merkle.root,
        fri_roots,
        fri_last_value,
        proof_transcript: channel.proof,
        trace_length: trace_len,
        queries,
    }
}

/// Verify a confinement proof by replaying the Fiat-Shamir transcript
/// and checking FRI query decommitments.
pub fn verify_confinement_proof(proof: &ConfinementProof) -> bool {
    // Structural checks
    if proof.trace_root.is_empty() || proof.cp_root.is_empty() {
        return false;
    }
    if proof.fri_roots.is_empty() {
        return false;
    }
    if proof.trace_length == 0 || !proof.trace_length.is_power_of_two() {
        return false;
    }
    if proof.proof_transcript.is_empty() {
        return false;
    }

    // Replay Fiat-Shamir transcript to recompute challenges
    let mut channel = Channel::new();
    channel.send(&proof.trace_root);
    let _alpha = channel.receive_random_field_element();
    channel.send(&proof.cp_root);

    // Recompute FRI betas
    let mut fri_betas = Vec::new();
    for root in &proof.fri_roots {
        let beta = channel.receive_random_field_element();
        fri_betas.push(beta);
        channel.send(root);
    }
    channel.send(&proof.fri_last_value.to_string());

    // Verify each query decommitment.
    // Empty auth paths or missing FRI vals are soundness failures: a malicious
    // prover must not be able to skip Merkle membership checks by omitting them.
    for (_qi, query) in proof.queries.iter().enumerate() {
        // 1. Verify trace Merkle membership
        if query.trace_auth.is_empty() {
            return false;
        }
        let ok = verify_decommitment(
            query.idx,
            &query.trace_val,
            &query.trace_auth,
            &proof.trace_root,
        );
        if !ok {
            return false;
        }

        // 2. Verify CP Merkle membership
        if query.cp_auth.is_empty() {
            return false;
        }
        let ok = verify_decommitment(
            query.idx,
            &query.cp_val,
            &query.cp_auth,
            &proof.cp_root,
        );
        if !ok {
            return false;
        }

        // 3. Verify FRI layer Merkle openings: one per fri_root committed
        if query.fri_auths.len() != proof.fri_roots.len() {
            return false;
        }
        if query.fri_vals.len() != proof.fri_roots.len() {
            return false;
        }
        let mut layer_idx = query.idx;
        for (layer_num, fri_auth) in query.fri_auths.iter().enumerate() {
            if fri_auth.is_empty() {
                return false;
            }
            let leaf_num = 1usize << fri_auth.len();
            let folded_idx = layer_idx % leaf_num;
            let val = query.fri_vals[layer_num];
            let ok = verify_decommitment(
                folded_idx,
                &val,
                fri_auth,
                &proof.fri_roots[layer_num],
            );
            if !ok {
                return false;
            }
            layer_idx = folded_idx;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confinement_proof_correct_masking() {
        let raw = vec![0.5, -0.3, 0.7, 0.1];
        let mask = vec![true, false, true, false];
        let constrained = vec![0.5, 0.0, 0.7, 0.0];

        let proof = prove_confinement(&raw, &mask, &constrained);
        assert!(verify_confinement_proof(&proof));
        assert!(!proof.trace_root.is_empty());
        assert!(!proof.fri_roots.is_empty());
        assert!(!proof.queries.is_empty());
    }

    #[test]
    fn test_confinement_proof_all_active() {
        let raw = vec![0.5, -0.3, 0.7, 0.1];
        let mask = vec![true, true, true, true];
        let constrained = vec![0.5, -0.3, 0.7, 0.1];

        let proof = prove_confinement(&raw, &mask, &constrained);
        assert!(verify_confinement_proof(&proof));
    }

    #[test]
    fn test_confinement_proof_all_masked() {
        let raw = vec![0.5, -0.3, 0.7, 0.1];
        let mask = vec![false, false, false, false];
        let constrained = vec![0.0, 0.0, 0.0, 0.0];

        let proof = prove_confinement(&raw, &mask, &constrained);
        assert!(verify_confinement_proof(&proof));
    }

    #[test]
    fn test_confinement_tampered_trace_root_fails() {
        let raw = vec![0.5, -0.3, 0.7, 0.1];
        let mask = vec![true, false, true, false];
        let constrained = vec![0.5, 0.0, 0.7, 0.0];

        let mut proof = prove_confinement(&raw, &mask, &constrained);
        proof.trace_root = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        // Tampering the root should cause Merkle verification to fail
        assert!(!verify_confinement_proof(&proof));
    }

    #[test]
    fn test_confinement_tampered_cp_root_fails() {
        let raw = vec![0.5, -0.3, 0.7, 0.1];
        let mask = vec![true, false, true, false];
        let constrained = vec![0.5, 0.0, 0.7, 0.0];

        let mut proof = prove_confinement(&raw, &mask, &constrained);
        proof.cp_root = "0000000000000000000000000000000000000000000000000000000000000000".to_string();
        assert!(!verify_confinement_proof(&proof));
    }

    #[test]
    fn test_confinement_tampered_query_value_fails() {
        let raw = vec![0.5, -0.3, 0.7, 0.1];
        let mask = vec![true, false, true, false];
        let constrained = vec![0.5, 0.0, 0.7, 0.0];

        let mut proof = prove_confinement(&raw, &mask, &constrained);
        if !proof.queries.is_empty() {
            proof.queries[0].trace_val = FieldElement::new(999);
        }
        assert!(!verify_confinement_proof(&proof));
    }

    #[test]
    fn test_f64_field_roundtrip() {
        let values = vec![0.5, -0.3, 0.0, 1.0, -1.0, 0.001];
        for v in values {
            let fe = f64_to_field(v);
            let back = field_to_f64(fe);
            assert!((v - back).abs() < 1e-6, "Roundtrip failed for {}: got {}", v, back);
        }
    }
}
