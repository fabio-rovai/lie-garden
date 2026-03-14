use crate::circuit::define::Circuit;
use crate::lie::group::GroupElement;

/// Compute reachable states from a starting point, given discrete algebra inputs.
/// Returns the set of reachable group elements (up to tolerance).
///
/// Bounded by max_depth (steps) and max_breadth (states per level).
pub fn compute_reachable(
    circuit: &Circuit,
    start: &GroupElement,
    sample_inputs: &[Vec<f64>],
    max_depth: usize,
    max_breadth: usize,
) -> Vec<GroupElement> {
    let mut visited: Vec<GroupElement> = vec![start.clone()];
    let mut frontier: Vec<GroupElement> = vec![start.clone()];

    for _depth in 0..max_depth {
        let mut next_frontier = Vec::new();
        for state in &frontier {
            for input in sample_inputs {
                let constrained = circuit.constrain_algebra(input);
                let step = circuit.group.exp(&constrained);
                let new_state = circuit.group.multiply(state, &step);

                // Check if we've seen this state (within tolerance)
                let is_new = !visited.iter().any(|v| v.close_to(&new_state, 1e-6));

                if is_new && visited.len() < max_breadth {
                    visited.push(new_state.clone());
                    next_frontier.push(new_state);
                }
            }
        }
        if next_frontier.is_empty() {
            break;
        }
        frontier = next_frontier;
    }

    visited
}
