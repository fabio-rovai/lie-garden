use crate::circuit::define::Circuit;
use crate::circuit::map::map_to_algebra;
use crate::lie::group::GroupElement;
use crate::crypto::stark::confinement::{prove_confinement, ConfinementProof};
use crate::crypto::stark::lineage::{prove_lineage, LineageProof};

/// Record of a single step's coefficients for batch proof generation.
#[derive(Clone)]
struct StepRecord {
    raw_coefficients: Vec<f64>,
    mask: Vec<bool>,
    constrained: Vec<f64>,
}

/// Audit proof bundle: confinement + lineage proofs for a run.
pub struct AuditProof {
    /// STARK proof that every step's masking was applied correctly
    pub confinement_proofs: Vec<ConfinementProof>,
    /// STARK proof that the event chain has integrity
    pub lineage_proof: Option<LineageProof>,
    /// Number of steps covered
    pub step_count: usize,
}

/// An agent running inside a circuit.
pub struct AgentRunner<'a> {
    circuit: &'a Circuit,
    state: GroupElement,
    step_count: usize,
    trajectory: Vec<GroupElement>,
    /// Accumulated step data for proof generation
    step_records: Vec<StepRecord>,
    /// Lineage event strings for hash chain proof
    lineage_events: Vec<String>,
    /// Session ID for lineage
    session_id: String,
}

impl<'a> AgentRunner<'a> {
    pub fn new(circuit: &'a Circuit) -> Self {
        let state = circuit.group.identity();
        Self {
            circuit,
            state: state.clone(),
            step_count: 0,
            trajectory: vec![state],
            step_records: Vec::new(),
            lineage_events: Vec::new(),
            session_id: crate::util::rand_id(),
        }
    }

    /// THE main operation. LLM output → algebra → exp → on-group.
    pub fn step(&mut self, llm_output: &str) -> GroupElement {
        // 1. Map text to algebra coefficients
        let raw_coeffs = map_to_algebra(llm_output, self.circuit.group.algebra_dim(), 1.0);

        // 2. Get the mask
        let mask = self.circuit.active_generators.clone()
            .unwrap_or_else(|| vec![true; self.circuit.group.algebra_dim()]);

        // 3. Apply circuit constraints (mask inactive generators)
        let constrained = self.circuit.constrain_algebra(&raw_coeffs);

        // 4. Exponential map — ALWAYS lands on group
        let step_element = self.circuit.group.exp(&constrained);

        // 5. Group multiplication — ALWAYS stays on group (closure axiom)
        self.state = self.circuit.group.multiply(&self.state, &step_element);

        self.step_count += 1;
        self.trajectory.push(self.state.clone());

        // 6. Record step data for proof generation
        self.step_records.push(StepRecord {
            raw_coefficients: raw_coeffs,
            mask,
            constrained,
        });

        // 7. Record lineage event
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let coeffs_summary = format!("dim{}", self.circuit.group.algebra_dim());
        let event = format!("{}:{}:{}:step:grav_step:{}",
            self.session_id, self.step_count, ts, coeffs_summary);
        self.lineage_events.push(event);

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

    /// Generate STARK audit proofs for all accumulated steps.
    /// Returns confinement proofs (one per step) and a lineage proof.
    pub fn audit_proof(&self) -> AuditProof {
        let confinement_proofs: Vec<ConfinementProof> = self.step_records.iter()
            .map(|record| {
                prove_confinement(&record.raw_coefficients, &record.mask, &record.constrained)
            })
            .collect();

        let lineage_proof = if !self.lineage_events.is_empty() {
            Some(prove_lineage(&self.lineage_events))
        } else {
            None
        };

        AuditProof {
            confinement_proofs,
            lineage_proof,
            step_count: self.step_count,
        }
    }

    /// Number of steps recorded for proof generation.
    pub fn pending_proof_steps(&self) -> usize {
        self.step_records.len()
    }

    pub fn eject(self) -> (GroupElement, Vec<GroupElement>) {
        (self.state, self.trajectory)
    }
}
