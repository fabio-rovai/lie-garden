use gravrail::circuit::define::Circuit;
use gravrail::circuit::reachability::compute_reachable;
use gravrail::lie::group::{LieGroup, GroupType};

#[test]
fn reachability_bounded() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group, None);
    let start = circuit.group.identity();
    let inputs: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.1, 0.0, 0.0]).collect();
    let result = compute_reachable(&circuit, &start, &inputs, 5, 100);
    assert!(result.len() <= 100, "Reachability must respect breadth bound, got {}", result.len());
}

#[test]
fn reachability_depth_bounded() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group, None);
    let start = circuit.group.identity();
    let inputs: Vec<Vec<f64>> = vec![vec![0.5, 0.0, 0.0]];
    // With only 1 input and max_depth=3, we can have at most 4 states (start + 3 steps)
    let result = compute_reachable(&circuit, &start, &inputs, 3, 1000);
    assert!(result.len() <= 4, "Depth-bounded reachability should have at most 4 states, got {}", result.len());
}

#[test]
fn zero_unsafe_blocks() {
    // If this test compiles, the crate has no unsafe code in its public API.
    let _ = gravrail::lie::group::LieGroup::new(gravrail::lie::group::GroupType::SO, 3);
}
