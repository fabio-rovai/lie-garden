use gravrail::circuit::define::Circuit;
use gravrail::circuit::reachability::compute_reachable;
use gravrail::lie::group::{LieGroup, GroupType};

// ── define.rs tests ──────────────────────────────────────────────────────────

#[test]
fn two_circuits_have_different_ids() {
    let group = LieGroup::new(GroupType::SO, 3);
    let c1 = Circuit::new(group.clone(), None);
    let c2 = Circuit::new(group.clone(), None);
    assert_ne!(c1.id, c2.id, "each Circuit::new() call must produce a unique ID");
}

#[test]
fn constrain_algebra_with_mask_zeroes_inactive_components() {
    let group = LieGroup::new(GroupType::SO, 3);
    // SO(3) has 3 algebra generators; mask: only first generator active
    let mask = vec![true, false, false];
    let circuit = Circuit::new(group, Some(mask));

    let coeffs = vec![1.0, 2.0, 3.0];
    let result = circuit.constrain_algebra(&coeffs);

    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.0).abs() < 1e-12, "active generator must be preserved");
    assert!((result[1]).abs() < 1e-12, "inactive generator must be zeroed");
    assert!((result[2]).abs() < 1e-12, "inactive generator must be zeroed");
}

#[test]
fn constrain_algebra_without_mask_passes_through_unchanged() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group, None);

    let coeffs = vec![1.5, -2.3, 0.7];
    let result = circuit.constrain_algebra(&coeffs);

    assert_eq!(result.len(), coeffs.len());
    for (a, b) in result.iter().zip(coeffs.iter()) {
        assert!((a - b).abs() < 1e-12, "no-mask circuit must pass coefficients through unchanged");
    }
}

#[test]
fn constrain_algebra_mask_shorter_than_coefficients_zeroes_overflow() {
    let group = LieGroup::new(GroupType::SO, 3);
    // mask only covers 2 of 3 generators; first two active
    let mask = vec![true, true];
    let circuit = Circuit::new(group, Some(mask));

    let coeffs = vec![1.0, 2.0, 3.0];
    let result = circuit.constrain_algebra(&coeffs);

    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.0).abs() < 1e-12);
    assert!((result[1] - 2.0).abs() < 1e-12);
    assert!((result[2]).abs() < 1e-12, "out-of-mask index must be zeroed");
}

#[test]
fn circuit_lock_sets_locked_flag() {
    let group = LieGroup::new(GroupType::SO, 3);
    let mut circuit = Circuit::new(group, None);
    assert!(!circuit.locked, "new circuit should not be locked");
    circuit.lock();
    assert!(circuit.locked, "locked() must set the locked flag");
}

// ── reachability.rs tests ─────────────────────────────────────────────────────

#[test]
fn identity_element_is_always_reachable() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group.clone(), None);

    let start = group.identity();
    // A single zero input: exp(0) = I, so the BFS will find the identity immediately
    let sample_inputs: Vec<Vec<f64>> = vec![vec![0.0, 0.0, 0.0]];

    let reachable = compute_reachable(&circuit, &start, &sample_inputs, 1, 100);

    // The start element (identity) must be in the reachable set
    assert!(
        reachable.iter().any(|e| e.close_to(&start, 1e-6)),
        "identity (start state) must appear in the reachable set"
    );
}

#[test]
fn reachable_set_includes_start_state() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group.clone(), None);

    let start = group.identity();
    let sample_inputs: Vec<Vec<f64>> = vec![vec![0.1, 0.0, 0.0]];

    let reachable = compute_reachable(&circuit, &start, &sample_inputs, 3, 50);

    assert!(
        !reachable.is_empty(),
        "reachable set must always contain at least the start state"
    );
    assert!(
        reachable.iter().any(|e| e.close_to(&start, 1e-6)),
        "start state must always be in the reachable set"
    );
}

#[test]
fn zero_depth_returns_only_start_state() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group.clone(), None);

    let start = group.identity();
    let sample_inputs: Vec<Vec<f64>> = vec![vec![1.0, 0.0, 0.0]];

    let reachable = compute_reachable(&circuit, &start, &sample_inputs, 0, 100);

    assert_eq!(reachable.len(), 1, "zero depth BFS must return exactly the start state");
    assert!(reachable[0].close_to(&start, 1e-6));
}

#[test]
fn max_breadth_limits_set_size() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group.clone(), None);

    let start = group.identity();
    // Varied inputs will explore many states
    let sample_inputs: Vec<Vec<f64>> = vec![
        vec![0.1, 0.0, 0.0],
        vec![0.0, 0.2, 0.0],
        vec![0.0, 0.0, 0.3],
    ];

    let max_breadth = 5;
    let reachable = compute_reachable(&circuit, &start, &sample_inputs, 10, max_breadth);

    assert!(
        reachable.len() <= max_breadth,
        "reachable set must not exceed max_breadth (got {} > {})",
        reachable.len(),
        max_breadth
    );
}

#[test]
fn empty_sample_inputs_returns_only_start() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group.clone(), None);

    let start = group.identity();
    let sample_inputs: Vec<Vec<f64>> = vec![];

    let reachable = compute_reachable(&circuit, &start, &sample_inputs, 5, 100);

    assert_eq!(
        reachable.len(),
        1,
        "with no inputs, BFS can only ever reach the start state"
    );
}
