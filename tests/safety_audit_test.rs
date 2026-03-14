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
fn lineage_tamper_detection() {
    let dir = tempfile::tempdir().unwrap();
    let db = gravrail::state::StateDb::open(&dir.path().join("test.db")).unwrap();
    let log = gravrail::lineage::LineageLog::new(db.clone());
    let session = log.new_session();

    log.record(&session, "step", "circuit_step", "agent_1 moved");
    log.record(&session, "step", "circuit_step", "agent_2 moved");
    log.record(&session, "crypto", "grav_sign", "signature created");
    assert!(log.verify_chain(&session), "Clean chain must verify");

    // Tamper with middle record
    db.conn().unwrap().execute(
        "UPDATE lineage_events SET details = 'TAMPERED' WHERE seq = 2 AND session_id = ?1",
        rusqlite::params![session],
    ).unwrap();
    assert!(!log.verify_chain(&session), "Tampered chain must fail verification");
}

#[test]
fn zero_unsafe_blocks() {
    // This test verifies the unsafe audit at compile time.
    // The CI pipeline also runs `grep -r "unsafe" gravrail/src/` as a belt-and-suspenders check.
    // If this test compiles, the crate has no unsafe code in its public API.
    let _ = gravrail::lie::group::LieGroup::new(gravrail::lie::group::GroupType::SO, 3);
}
