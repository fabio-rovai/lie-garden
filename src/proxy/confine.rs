// Per-response confinement pipeline

use std::fmt;

use crate::circuit::define::Circuit;
use crate::circuit::map::map_to_algebra;
use crate::lie::group::GroupElement;
use crate::crypto::stark::confinement::{prove_confinement, ConfinementProof};

/// Result of running text through the confinement pipeline.
#[derive(Clone)]
pub struct ConfinementResult {
    /// The original text that was confined.
    pub text: String,
    /// Current group state after this step.
    pub state: GroupElement,
    /// Sequence number of this step.
    pub seq: u64,
    /// Optional STARK proof of correct constraint masking.
    pub proof: Option<ConfinementProof>,
    /// Flat representation of the state matrix elements.
    pub state_elements: Vec<f64>,
}

impl fmt::Debug for ConfinementResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConfinementResult")
            .field("text", &self.text)
            .field("state", &self.state)
            .field("seq", &self.seq)
            .field("proof", &self.proof.as_ref().map(|_| "..."))
            .field("state_elements", &self.state_elements)
            .finish()
    }
}

/// Errors from the confinement pipeline.
#[derive(Debug, Clone)]
pub enum ConfinementError {
    /// The new state exceeds the reachability bound.
    ReachabilityViolation {
        text_preview: String,
        state: GroupElement,
    },
    /// Generic pipeline error.
    PipelineError(String),
}

impl fmt::Display for ConfinementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfinementError::ReachabilityViolation { text_preview, state } => {
                write!(
                    f,
                    "Reachability violation: text '{}' would move state (dim={}) beyond bound",
                    text_preview,
                    state.dim()
                )
            }
            ConfinementError::PipelineError(msg) => {
                write!(f, "Pipeline error: {}", msg)
            }
        }
    }
}

impl std::error::Error for ConfinementError {}

/// Compute the Frobenius distance between two group elements.
pub fn state_frobenius_distance(a: &GroupElement, b: &GroupElement) -> f64 {
    let a_data = a.matrix();
    let b_data = b.matrix();
    a_data
        .iter()
        .zip(b_data.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The confinement pipeline. Every LLM response runs through this.
pub struct ConfinementPipeline {
    circuit: Circuit,
    state: GroupElement,
    seq: u64,
    max_state_norm: Option<f64>,
    prove_every_step: bool,
}

impl ConfinementPipeline {
    /// Create a new pipeline starting at the identity element.
    pub fn new(circuit: Circuit, max_state_norm: Option<f64>, prove_every_step: bool) -> Self {
        let state = circuit.group.identity();
        Self {
            circuit,
            state,
            seq: 0,
            max_state_norm,
            prove_every_step,
        }
    }

    /// Confine a text response through the pipeline.
    ///
    /// Steps:
    /// 1. Map text to algebra coefficients
    /// 2. Constrain via active generator mask
    /// 3. Exponentiate to group element
    /// 4. Multiply into running state
    /// 5. Check reachability bound
    /// 6. Optionally generate STARK proof
    pub fn confine(&mut self, text: &str) -> Result<ConfinementResult, ConfinementError> {
        let algebra_dim = self.circuit.group.algebra_dim();

        // 1. Map text to algebra coefficients
        let raw_coeffs = map_to_algebra(text, algebra_dim, 1.0);

        // 2. Constrain to active generators
        let constrained = self.circuit.constrain_algebra(&raw_coeffs);

        // 3. Exponentiate: algebra → group
        let step_element = self.circuit.group.exp(&constrained);

        // 4. Multiply into running state
        let new_state = self.circuit.group.multiply(&self.state, &step_element);

        // 5. Reachability check
        if let Some(bound) = self.max_state_norm {
            let identity = self.circuit.group.identity();
            let dist = state_frobenius_distance(&identity, &new_state);
            if dist > bound {
                let preview = if text.len() > 50 {
                    format!("{}...", &text[..50])
                } else {
                    text.to_string()
                };
                return Err(ConfinementError::ReachabilityViolation {
                    text_preview: preview,
                    state: new_state,
                });
            }
        }

        // 6. Update state and sequence
        self.state = new_state.clone();
        self.seq += 1;

        // 7. Optionally generate STARK proof
        let proof = if self.prove_every_step {
            let mask = match &self.circuit.active_generators {
                Some(m) => {
                    let mut mask = m.clone();
                    mask.resize(algebra_dim, false);
                    mask
                }
                None => vec![true; algebra_dim],
            };
            Some(prove_confinement(&raw_coeffs, &mask, &constrained))
        } else {
            None
        };

        let state_elements = self.state.matrix().to_vec();

        Ok(ConfinementResult {
            text: text.to_string(),
            state: self.state.clone(),
            seq: self.seq,
            proof,
            state_elements,
        })
    }

    /// Current group state.
    pub fn state(&self) -> &GroupElement {
        &self.state
    }

    /// Current sequence number.
    pub fn seq(&self) -> u64 {
        self.seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lie::group::{LieGroup, GroupType};

    fn so3_circuit(active: Option<Vec<bool>>) -> Circuit {
        let group = LieGroup::new(GroupType::SO, 3);
        Circuit::new(group, active)
    }

    #[test]
    fn test_confine_basic() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let result = pipeline.confine("hello world");
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.seq, 1);
        assert_eq!(r.text, "hello world");
    }

    #[test]
    fn test_confine_sequence_increments() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        pipeline.confine("step one").unwrap();
        pipeline.confine("step two").unwrap();
        let r = pipeline.confine("step three").unwrap();
        assert_eq!(r.seq, 3);
        assert_eq!(pipeline.seq(), 3);
    }

    #[test]
    fn test_confine_state_changes() {
        let circuit = so3_circuit(None);
        let identity = circuit.group.identity();
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let r = pipeline.confine("move somewhere").unwrap();
        // State should differ from identity after a step
        assert!(!r.state.close_to(&identity, 1e-10));
    }

    #[test]
    fn test_confine_deterministic() {
        let circuit1 = so3_circuit(None);
        let circuit2 = so3_circuit(None);
        let mut p1 = ConfinementPipeline::new(circuit1, None, false);
        let mut p2 = ConfinementPipeline::new(circuit2, None, false);
        let r1 = p1.confine("same input").unwrap();
        let r2 = p2.confine("same input").unwrap();
        assert!(r1.state.close_to(&r2.state, 1e-12));
    }

    #[test]
    fn test_confine_with_reachability_bound() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, Some(0.001), false);
        let result = pipeline.confine("this should violate the tiny bound");
        assert!(result.is_err());
        match result.unwrap_err() {
            ConfinementError::ReachabilityViolation { text_preview, state: _ } => {
                assert!(!text_preview.is_empty());
            }
            _ => panic!("Expected ReachabilityViolation"),
        }
    }

    #[test]
    fn test_confine_generous_bound_passes() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, Some(100.0), false);
        let result = pipeline.confine("this should be fine with a generous bound");
        assert!(result.is_ok());
    }

    #[test]
    fn test_confine_with_proof() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, None, true);
        let r = pipeline.confine("prove this step").unwrap();
        assert!(r.proof.is_some());
    }

    #[test]
    fn test_confine_without_proof() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let r = pipeline.confine("no proof needed").unwrap();
        assert!(r.proof.is_none());
    }

    #[test]
    fn test_confine_with_masked_generators() {
        let circuit = so3_circuit(Some(vec![true, false, false]));
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let result = pipeline.confine("masked confinement");
        assert!(result.is_ok());
    }

    #[test]
    fn test_state_elements_in_result() {
        let circuit = so3_circuit(None);
        let mut pipeline = ConfinementPipeline::new(circuit, None, false);
        let r = pipeline.confine("check state elements").unwrap();
        // SO(3) has 3x3 = 9 matrix elements
        assert_eq!(r.state_elements.len(), 9);
    }
}
