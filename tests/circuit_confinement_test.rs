use gravrail::circuit::define::Circuit;
use gravrail::circuit::agent::AgentRunner;
use gravrail::lie::group::{LieGroup, GroupType};
use proptest::prelude::*;

proptest! {
    #[test]
    fn every_step_is_on_group(
        input in "[a-zA-Z0-9 ]{1,100}",
    ) {
        let group = LieGroup::new(GroupType::SO, 3);
        let circuit = Circuit::new(group.clone(), None);
        let mut runner = AgentRunner::new(&circuit);

        let result = runner.step(&input);
        prop_assert!(group.is_member(&result), "Step output must be on-group for input: {}", input);
    }

    #[test]
    fn adversarial_inputs_still_on_group(
        bytes in prop::collection::vec(0u8..255, 1..500),
    ) {
        let group = LieGroup::new(GroupType::SO, 3);
        let circuit = Circuit::new(group.clone(), None);
        let mut runner = AgentRunner::new(&circuit);

        let garbage = String::from_utf8_lossy(&bytes).to_string();
        let result = runner.step(&garbage);
        prop_assert!(group.is_member(&result), "Even garbage input must produce on-group output");
    }

    #[test]
    fn multi_step_stays_on_group(
        inputs in prop::collection::vec("[a-zA-Z ]{1,50}", 1..20),
    ) {
        let group = LieGroup::new(GroupType::SO, 3);
        let circuit = Circuit::new(group.clone(), None);
        let mut runner = AgentRunner::new(&circuit);

        for input in &inputs {
            let result = runner.step(input);
            prop_assert!(group.is_member(&result), "Every step must be on-group");
        }
    }
}
