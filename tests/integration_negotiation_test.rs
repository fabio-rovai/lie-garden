use gravrail::circuit::define::Circuit;
use gravrail::circuit::agent::AgentRunner;
use gravrail::lie::group::{LieGroup, GroupType};

#[test]
fn two_agent_negotiation_all_on_group() {
    let group = LieGroup::new(GroupType::SO, 3);
    let circuit = Circuit::new(group.clone(), None);
    let mut alice = AgentRunner::new(&circuit);
    let mut bob = AgentRunner::new(&circuit);

    let alice_moves = ["I propose we split 60/40", "Counter: 55/45", "Agreed at 55/45"];
    let bob_moves = ["I want 50/50", "55/45 is acceptable", "Confirmed"];

    for (a_move, b_move) in alice_moves.iter().zip(bob_moves.iter()) {
        let a_state = alice.step(a_move);
        let b_state = bob.step(b_move);
        assert!(group.is_member(&a_state), "Alice must stay on-group");
        assert!(group.is_member(&b_state), "Bob must stay on-group");
    }

    assert_eq!(alice.step_count(), 3);
    assert_eq!(bob.step_count(), 3);
}

#[test]
fn constrained_circuit_masks_generators() {
    let group = LieGroup::new(GroupType::SO, 3);
    // Only allow rotation around z-axis (generator index 2)
    let mask = vec![false, false, true];
    let circuit = Circuit::new(group.clone(), Some(mask));
    let mut runner = AgentRunner::new(&circuit);

    for msg in &["rotate left", "rotate right", "spin around"] {
        let state = runner.step(msg);
        assert!(group.is_member(&state), "Constrained steps must stay on-group");
    }
}
