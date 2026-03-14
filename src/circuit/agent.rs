use crate::circuit::define::Circuit;
use crate::circuit::map::map_to_algebra;
use crate::lie::group::GroupElement;

/// An agent running inside a circuit.
pub struct AgentRunner<'a> {
    circuit: &'a Circuit,
    state: GroupElement,
    step_count: usize,
    trajectory: Vec<GroupElement>,
}

impl<'a> AgentRunner<'a> {
    pub fn new(circuit: &'a Circuit) -> Self {
        let state = circuit.group.identity();
        Self {
            circuit,
            state: state.clone(),
            step_count: 0,
            trajectory: vec![state],
        }
    }

    /// THE main operation. LLM output → algebra → exp → on-group.
    pub fn step(&mut self, llm_output: &str) -> GroupElement {
        // 1. Map text to algebra coefficients
        let raw_coeffs = map_to_algebra(llm_output, self.circuit.group.algebra_dim(), 1.0);

        // 2. Apply circuit constraints (mask inactive generators)
        let constrained = self.circuit.constrain_algebra(&raw_coeffs);

        // 3. Exponential map — ALWAYS lands on group
        let step_element = self.circuit.group.exp(&constrained);

        // 4. Group multiplication — ALWAYS stays on group (closure axiom)
        self.state = self.circuit.group.multiply(&self.state, &step_element);

        self.step_count += 1;
        self.trajectory.push(self.state.clone());

        self.state.clone()
    }

    pub fn state(&self) -> &GroupElement {
        &self.state
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn trajectory(&self) -> &[GroupElement] {
        &self.trajectory
    }

    pub fn eject(self) -> (GroupElement, Vec<GroupElement>) {
        (self.state, self.trajectory)
    }
}
