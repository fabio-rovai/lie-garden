use rmcp::{
    ServerHandler, tool, tool_router, tool_handler,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo, Tool},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::circuit::define::Circuit;
use crate::circuit::map::map_to_algebra;
use crate::circuit::reachability::compute_reachable;
use crate::crypto::{commit, keyexchange, sign, zkproof};
use crate::gauge::bundle::FiberBundle;
use crate::gauge::connection::Connection;
use crate::gauge::curvature;
use crate::gauge::invariance::check_gauge_invariance;
use crate::inputs::*;
use crate::lie::algebra::LieAlgebra;
use crate::lie::group::{GroupElement, GroupType, LieGroup};
use crate::lie::invariants;
use crate::lineage::LineageLog;
use crate::state::StateDb;
use crate::util;

/// In-memory agent state (owns circuit reference + current position).
struct AgentState {
    state: GroupElement,
    step_count: usize,
    trajectory: Vec<GroupElement>,
    circuit_id: String,
}

#[derive(Clone)]
pub struct GravRailServer {
    tool_router: ToolRouter<Self>,
    db: StateDb,
    session_id: String,
    circuits: Arc<Mutex<HashMap<String, Circuit>>>,
    agents: Arc<Mutex<HashMap<String, AgentState>>>,
}

impl GravRailServer {
    pub fn new(db: StateDb) -> Self {
        let lineage = LineageLog::new(db.clone());
        let session_id = lineage.new_session();
        Self {
            tool_router: Self::tool_router(),
            db,
            session_id,
            circuits: Arc::new(Mutex::new(HashMap::new())),
            agents: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn list_tool_definitions(&self) -> Vec<Tool> {
        self.tool_router.list_all()
    }

    fn parse_group_type(s: &str) -> Result<GroupType, String> {
        match s.to_uppercase().as_str() {
            "SO" => Ok(GroupType::SO),
            "SE" => Ok(GroupType::SE),
            "GL" => Ok(GroupType::GL),
            _ => Err(format!("Unknown group type: {}. Use SO, SE, or GL.", s)),
        }
    }

    fn make_group(group_type: &str, n: usize) -> Result<LieGroup, String> {
        if n == 0 {
            return Err("Dimension n must be >= 1".to_string());
        }
        let gt = Self::parse_group_type(group_type)?;
        if gt == GroupType::SE && n < 2 {
            return Err("SE(n) requires n >= 2".to_string());
        }
        Ok(LieGroup::new(gt, n))
    }

    fn element_from_data(data: &[f64], n: usize) -> Result<GroupElement, String> {
        if data.len() != n * n {
            return Err(format!("Expected {} matrix entries for n={}, got {}", n * n, n, data.len()));
        }
        let m = nalgebra::DMatrix::from_column_slice(n, n, data);
        Ok(GroupElement::from_matrix(&m))
    }

    fn lineage(&self) -> LineageLog {
        LineageLog::new(self.db.clone())
    }

    fn log_event(&self, event_type: &str, operation: &str, details: &str) {
        self.lineage().record(&self.session_id, event_type, operation, details);
    }
}

#[tool_router]
impl GravRailServer {
    // ========== Status ==========

    #[tool(name = "grav_status", description = "Returns GravRail server health status and version")]
    fn grav_status(&self) -> String {
        let circuit_count = self.circuits.lock().map(|c| c.len()).unwrap_or(0);
        let agent_count = self.agents.lock().map(|a| a.len()).unwrap_or(0);
        serde_json::json!({
            "status": "ok",
            "version": env!("CARGO_PKG_VERSION"),
            "session_id": self.session_id,
            "circuits": circuit_count,
            "agents": agent_count,
        }).to_string()
    }

    // ========== Lie Group Tools ==========

    #[tool(name = "grav_exp", description = "Exponential map: algebra coefficients → on-group element. THE core confinement guarantee.")]
    fn grav_exp(&self, Parameters(input): Parameters<GravExpInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let elem = g.exp(&input.coefficients);
                self.log_event("compute", "grav_exp", &format!("{}({})", input.group_type, input.n));
                serde_json::json!({"matrix": elem.matrix(), "n": elem.dim(), "on_group": g.is_member(&elem)}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_log", description = "Logarithmic map: group element → algebra coefficients")]
    fn grav_log(&self, Parameters(input): Parameters<GravLogInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => match Self::element_from_data(&input.matrix, input.n) {
                Ok(elem) => {
                    let coeffs = g.log(&elem);
                    serde_json::json!({"coefficients": coeffs}).to_string()
                }
                Err(e) => serde_json::json!({"error": e}).to_string(),
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_multiply", description = "Group multiplication: a * b → on-group result")]
    fn grav_multiply(&self, Parameters(input): Parameters<GravMultiplyInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let a = match Self::element_from_data(&input.a, input.n) { Ok(v) => v, Err(e) => return serde_json::json!({"error": e}).to_string() };
                let b = match Self::element_from_data(&input.b, input.n) { Ok(v) => v, Err(e) => return serde_json::json!({"error": e}).to_string() };
                let result = g.multiply(&a, &b);
                serde_json::json!({"matrix": result.matrix(), "on_group": g.is_member(&result)}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_inverse", description = "Group inverse of an element")]
    fn grav_inverse(&self, Parameters(input): Parameters<GravInverseInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => match Self::element_from_data(&input.matrix, input.n) {
                Ok(elem) => {
                    let inv = g.inverse(&elem);
                    serde_json::json!({"matrix": inv.matrix()}).to_string()
                }
                Err(e) => serde_json::json!({"error": e}).to_string(),
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_identity", description = "Identity element for the specified group")]
    fn grav_identity(&self, Parameters(input): Parameters<GravIdentityInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let id = g.identity();
                serde_json::json!({"matrix": id.matrix()}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_is_member", description = "Check if a matrix is a valid group element")]
    fn grav_is_member(&self, Parameters(input): Parameters<GravIsMemberInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => match Self::element_from_data(&input.matrix, input.n) {
                Ok(elem) => serde_json::json!({"is_member": g.is_member(&elem)}).to_string(),
                Err(e) => serde_json::json!({"error": e}).to_string(),
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_project", description = "Project a matrix onto the group manifold (nearest valid element)")]
    fn grav_project(&self, Parameters(input): Parameters<GravProjectInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => match Self::element_from_data(&input.matrix, input.n) {
                Ok(elem) => {
                    let projected = g.project_to_group(&elem);
                    serde_json::json!({"matrix": projected.matrix(), "on_group": g.is_member(&projected)}).to_string()
                }
                Err(e) => serde_json::json!({"error": e}).to_string(),
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    // ========== Lie Algebra Tools ==========

    #[tool(name = "grav_bracket", description = "Lie bracket [X, Y] of two algebra elements")]
    fn grav_bracket(&self, Parameters(input): Parameters<GravBracketInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let alg = LieAlgebra::new(g.group_type, g.n);
                let result = alg.bracket(&input.x, &input.y);
                serde_json::json!({"bracket": result}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_killing_form", description = "Killing form B(X, Y) = tr(ad(X) * ad(Y))")]
    fn grav_killing_form(&self, Parameters(input): Parameters<GravKillingFormInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let alg = LieAlgebra::new(g.group_type, g.n);
                let kf = invariants::killing_form(&alg, &input.x, &input.y);
                serde_json::json!({"killing_form": kf}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_algebra_info", description = "Get algebra dimension, structure constants, semisimplicity for a group")]
    fn grav_algebra_info(&self, Parameters(input): Parameters<GravAlgebraInfoInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let alg = LieAlgebra::new(g.group_type, g.n);
                let dim = alg.dim();
                let semisimple = invariants::is_semisimple(&alg);
                let km = invariants::killing_matrix(&alg);
                serde_json::json!({
                    "algebra_dim": dim,
                    "matrix_dim": g.matrix_dim(),
                    "is_semisimple": semisimple,
                    "killing_matrix": km,
                }).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    // ========== Gauge Tools ==========

    #[tool(name = "grav_holonomy", description = "Compute holonomy (parallel transport around a closed path) — detects curvature/manipulation")]
    fn grav_holonomy(&self, Parameters(input): Parameters<GravHolonomyInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let bundle = FiberBundle::new(g.clone(), input.base_dim);
                let conn = if input.curvature.abs() < 1e-10 {
                    Connection::flat(&bundle)
                } else {
                    Connection::with_curvature(&bundle, input.curvature)
                };
                let h = conn.holonomy(&input.path);
                let id = g.identity();
                let deviation: f64 = h.matrix().iter().zip(id.matrix().iter())
                    .map(|(a, b)| (a - b).abs()).sum();
                self.log_event("compute", "grav_holonomy", &format!("deviation={:.6}", deviation));
                serde_json::json!({
                    "holonomy_matrix": h.matrix(),
                    "deviation_from_identity": deviation,
                    "is_trivial": deviation < 1e-6,
                }).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_gauge_invariance", description = "Check if trace observable is gauge-invariant under conjugation")]
    fn grav_gauge_invariance(&self, Parameters(input): Parameters<GravGaugeInvarianceInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => match Self::element_from_data(&input.state, input.n) {
                Ok(state) => {
                    let gauge_elem = g.exp(&input.gauge_coefficients);
                    let diff = check_gauge_invariance(&g, &state, &gauge_elem, |e| {
                        e.to_matrix().trace()
                    });
                    serde_json::json!({
                        "invariance_violation": diff,
                        "is_invariant": diff < 1e-8,
                    }).to_string()
                }
                Err(e) => serde_json::json!({"error": e}).to_string(),
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_curvature", description = "Compute field strength tensor F_{μν} for a connection")]
    fn grav_curvature(&self, Parameters(input): Parameters<GravCurvatureInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let bundle = FiberBundle::new(g, input.base_dim);
                let conn = Connection::with_curvature(&bundle, input.curvature);
                let is_flat = curvature::is_flat(&conn);
                let mut field_strengths = Vec::new();
                for mu in 0..input.base_dim {
                    for nu in (mu+1)..input.base_dim {
                        let fs = curvature::field_strength(&conn, mu, nu);
                        field_strengths.push(serde_json::json!({"mu": mu, "nu": nu, "F": fs}));
                    }
                }
                serde_json::json!({
                    "is_flat": is_flat,
                    "field_strengths": field_strengths,
                }).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    // ========== Crypto Tools ==========

    #[tool(name = "grav_commit", description = "Pedersen commitment on a Lie group: C = exp(x) * exp_h(r)")]
    fn grav_commit(&self, Parameters(input): Parameters<GravCommitInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let c = commit::commit(&g, &input.value, &input.randomness);
                self.log_event("crypto", "grav_commit", "commitment created");
                serde_json::json!({"commitment": c.matrix()}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_verify_commit", description = "Verify a Pedersen commitment opening")]
    fn grav_verify_commit(&self, Parameters(input): Parameters<GravVerifyCommitInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => match Self::element_from_data(&input.commitment, input.n) {
                Ok(c) => {
                    let valid = commit::verify_commit(&g, &c, &input.value, &input.randomness);
                    serde_json::json!({"valid": valid}).to_string()
                }
                Err(e) => serde_json::json!({"error": e}).to_string(),
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_sign", description = "Schnorr signature on a Lie group")]
    fn grav_sign(&self, Parameters(input): Parameters<GravSignInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let sig = sign::sign(&g, input.message.as_bytes(), &input.secret);
                self.log_event("crypto", "grav_sign", "signature created");
                serde_json::json!({"r": sig.r.matrix(), "s": sig.s}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_verify_sign", description = "Verify a Schnorr signature")]
    fn grav_verify_sign(&self, Parameters(input): Parameters<GravVerifySignInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let public = match Self::element_from_data(&input.public_key, input.n) { Ok(v) => v, Err(e) => return serde_json::json!({"error": e}).to_string() };
                let r = match Self::element_from_data(&input.sig_r, input.n) { Ok(v) => v, Err(e) => return serde_json::json!({"error": e}).to_string() };
                let sig = sign::Signature { r, s: input.sig_s };
                let valid = sign::verify(&g, input.message.as_bytes(), &public, &sig);
                serde_json::json!({"valid": valid}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_dh", description = "Key agreement on a Lie group (algebra-space shared secret)")]
    fn grav_dh(&self, Parameters(input): Parameters<GravDiffieHellmanInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let (pub_a, pub_b, shared_a, _shared_b) = keyexchange::diffie_hellman(&g, &input.secret_a, &input.secret_b);
                self.log_event("crypto", "grav_dh", "key exchange completed");
                serde_json::json!({
                    "public_a": pub_a.matrix(),
                    "public_b": pub_b.matrix(),
                    "shared_secret": shared_a.matrix(),
                }).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_zk_prove", description = "Generate ZK proof of knowledge of a group element's discrete log")]
    fn grav_zk_prove(&self, Parameters(input): Parameters<GravZkProveInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let public = g.exp(&input.secret);
                let proof = zkproof::prove_knowledge(&g, &input.secret, &public);
                self.log_event("crypto", "grav_zk_prove", "ZK proof generated");
                serde_json::json!({
                    "public_key": public.matrix(),
                    "commitment": proof.commitment.matrix(),
                    "response": proof.response,
                }).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_zk_verify", description = "Verify a ZK proof of discrete log knowledge")]
    fn grav_zk_verify(&self, Parameters(input): Parameters<GravZkVerifyInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let public = match Self::element_from_data(&input.public_key, input.n) { Ok(v) => v, Err(e) => return serde_json::json!({"error": e}).to_string() };
                let commitment = match Self::element_from_data(&input.proof_commitment, input.n) { Ok(v) => v, Err(e) => return serde_json::json!({"error": e}).to_string() };
                let proof = zkproof::ZkProof { commitment, response: input.proof_response };
                let valid = zkproof::verify_knowledge(&g, &public, &proof);
                serde_json::json!({"valid": valid}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    // ========== Circuit Tools ==========

    #[tool(name = "grav_circuit_create", description = "Create a new confinement circuit with a Lie group")]
    fn grav_circuit_create(&self, Parameters(input): Parameters<GravCircuitCreateInput>) -> String {
        match Self::make_group(&input.group_type, input.n) {
            Ok(g) => {
                let circuit = Circuit::new(g, input.active_generators);
                let id = circuit.id.clone();
                if let Ok(mut circuits) = self.circuits.lock() {
                    circuits.insert(id.clone(), circuit);
                }
                self.log_event("circuit", "grav_circuit_create", &id);
                serde_json::json!({"circuit_id": id, "group_type": input.group_type, "n": input.n}).to_string()
            }
            Err(e) => serde_json::json!({"error": e}).to_string(),
        }
    }

    #[tool(name = "grav_circuit_list", description = "List all active circuits")]
    fn grav_circuit_list(&self, Parameters(input): Parameters<GravCircuitListInput>) -> String {
        let circuits = match self.circuits.lock() {
            Ok(c) => c,
            Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
        };
        let list: Vec<_> = circuits.iter()
            .filter(|(_id, c)| {
                input.group_type.as_ref().is_none_or(|gt| {
                    format!("{:?}", c.group.group_type).to_uppercase() == gt.to_uppercase()
                })
            })
            .map(|(id, c)| serde_json::json!({
                "id": id,
                "group_type": format!("{:?}", c.group.group_type),
                "n": c.group.n,
                "locked": c.locked,
            }))
            .collect();
        serde_json::json!({"circuits": list}).to_string()
    }

    #[tool(name = "grav_circuit_lock", description = "Lock a circuit (prevent modification)")]
    fn grav_circuit_lock(&self, Parameters(input): Parameters<GravCircuitLockInput>) -> String {
        let mut circuits = match self.circuits.lock() {
            Ok(c) => c,
            Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
        };
        match circuits.get_mut(&input.circuit_id) {
            Some(c) => {
                c.lock();
                self.log_event("circuit", "grav_circuit_lock", &input.circuit_id);
                serde_json::json!({"locked": true, "circuit_id": input.circuit_id}).to_string()
            }
            None => serde_json::json!({"error": format!("Circuit not found: {}", input.circuit_id)}).to_string(),
        }
    }

    #[tool(name = "grav_agent_create", description = "Spawn a new agent inside a circuit (starts at identity)")]
    fn grav_agent_create(&self, Parameters(input): Parameters<GravCircuitAgentCreateInput>) -> String {
        let circuits = match self.circuits.lock() {
            Ok(c) => c,
            Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
        };
        match circuits.get(&input.circuit_id) {
            Some(circuit) => {
                let agent_id = util::rand_id();
                let state = circuit.group.identity();
                if let Ok(mut agents) = self.agents.lock() {
                    agents.insert(agent_id.clone(), AgentState {
                        state: state.clone(),
                        step_count: 0,
                        trajectory: vec![state],
                        circuit_id: input.circuit_id.clone(),
                    });
                }
                self.log_event("agent", "grav_agent_create", &format!("{}:{}", input.circuit_id, agent_id));
                serde_json::json!({"agent_id": agent_id, "circuit_id": input.circuit_id}).to_string()
            }
            None => serde_json::json!({"error": format!("Circuit not found: {}", input.circuit_id)}).to_string(),
        }
    }

    #[tool(name = "grav_step", description = "THE core operation. LLM output text → algebra → exp → on-group state update.")]
    fn grav_step(&self, Parameters(input): Parameters<GravCircuitStepInput>) -> String {
        // Clone circuit data to release the lock before computing
        let circuit = {
            let circuits = match self.circuits.lock() {
                Ok(c) => c,
                Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
            };
            match circuits.get(&input.circuit_id) {
                Some(c) => c.clone(),
                None => return serde_json::json!({"error": format!("Circuit not found: {}", input.circuit_id)}).to_string(),
            }
        };

        // Compute outside locks
        let raw_coeffs = map_to_algebra(&input.text, circuit.group.algebra_dim(), 1.0);
        let constrained = circuit.constrain_algebra(&raw_coeffs);
        let step_elem = circuit.group.exp(&constrained);

        // Update agent state under lock
        let mut agents = match self.agents.lock() {
            Ok(a) => a,
            Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
        };
        let agent = match agents.get_mut(&input.agent_id) {
            Some(a) => a,
            None => return serde_json::json!({"error": format!("Agent not found: {}", input.agent_id)}).to_string(),
        };

        agent.state = circuit.group.multiply(&agent.state, &step_elem);
        agent.step_count += 1;
        agent.trajectory.push(agent.state.clone());

        let on_group = circuit.group.is_member(&agent.state);
        let step_count = agent.step_count;
        let state_matrix = agent.state.matrix().to_vec();
        drop(agents);

        self.log_event("step", "grav_step", &format!("{}:{} step={}", input.circuit_id, input.agent_id, step_count));

        serde_json::json!({
            "state": state_matrix,
            "step_count": step_count,
            "on_group": on_group,
            "algebra_coefficients": constrained,
        }).to_string()
    }

    #[tool(name = "grav_agent_state", description = "Get current state of an agent")]
    fn grav_agent_state(&self, Parameters(input): Parameters<GravCircuitAgentStateInput>) -> String {
        let agents = match self.agents.lock() {
            Ok(a) => a,
            Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
        };
        match agents.get(&input.agent_id) {
            Some(a) => {
                let circuits = match self.circuits.lock() {
                    Ok(c) => c,
                    Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
                };
                let on_group = circuits.get(&a.circuit_id)
                    .map(|c| c.group.is_member(&a.state))
                    .unwrap_or(false);
                serde_json::json!({
                    "state": a.state.matrix(),
                    "step_count": a.step_count,
                    "circuit_id": a.circuit_id,
                    "on_group": on_group,
                }).to_string()
            }
            None => serde_json::json!({"error": format!("Agent not found: {}", input.agent_id)}).to_string(),
        }
    }

    #[tool(name = "grav_trajectory", description = "Get full trajectory of an agent")]
    fn grav_trajectory(&self, Parameters(input): Parameters<GravTrajectoryInput>) -> String {
        let agents = match self.agents.lock() {
            Ok(a) => a,
            Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
        };
        match agents.get(&input.agent_id) {
            Some(a) => {
                let points: Vec<_> = a.trajectory.iter()
                    .map(|e| e.matrix().to_vec())
                    .collect();
                serde_json::json!({"trajectory": points, "steps": a.step_count}).to_string()
            }
            None => serde_json::json!({"error": format!("Agent not found: {}", input.agent_id)}).to_string(),
        }
    }

    #[tool(name = "grav_map_to_algebra", description = "Map arbitrary text to Lie algebra coefficients (the chokepoint function)")]
    fn grav_map_to_algebra(&self, Parameters(input): Parameters<GravMapToAlgebraInput>) -> String {
        let scale = input.scale.unwrap_or(1.0);
        let coeffs = map_to_algebra(&input.text, input.algebra_dim, scale);
        serde_json::json!({"coefficients": coeffs}).to_string()
    }

    #[tool(name = "grav_reachability", description = "Compute reachable states from current position (bounded BFS)")]
    fn grav_reachability(&self, Parameters(input): Parameters<GravReachabilityInput>) -> String {
        let circuit = {
            let circuits = match self.circuits.lock() {
                Ok(c) => c,
                Err(_) => return serde_json::json!({"error": "lock poisoned"}).to_string(),
            };
            match circuits.get(&input.circuit_id) {
                Some(c) => c.clone(),
                None => return serde_json::json!({"error": format!("Circuit not found: {}", input.circuit_id)}).to_string(),
            }
        };
        let start = circuit.group.identity();
        let reachable = compute_reachable(&circuit, &start, &input.sample_inputs, input.max_depth, input.max_breadth);
        self.log_event("compute", "grav_reachability", &format!("{} states", reachable.len()));
        serde_json::json!({"reachable_count": reachable.len(), "max_depth": input.max_depth, "max_breadth": input.max_breadth}).to_string()
    }

    // ========== Lifecycle Tools ==========

    #[tool(name = "grav_lineage", description = "View the session's lineage trail")]
    fn grav_lineage(&self, Parameters(input): Parameters<GravLineageInput>) -> String {
        let sid = input.session_id.as_deref().unwrap_or(&self.session_id);
        let compact = self.lineage().get_compact(sid);
        serde_json::json!({"session_id": sid, "lineage": compact}).to_string()
    }

    #[tool(name = "grav_lineage_verify", description = "Verify lineage hash chain integrity (tamper detection)")]
    fn grav_lineage_verify(&self, Parameters(input): Parameters<GravLineageVerifyInput>) -> String {
        let valid = self.lineage().verify_chain(&input.session_id);
        serde_json::json!({"session_id": input.session_id, "chain_valid": valid}).to_string()
    }

    #[tool(name = "grav_feedback", description = "Record tool feedback (accept/dismiss) for adaptive behavior")]
    fn grav_feedback(&self, Parameters(input): Parameters<GravFeedbackInput>) -> String {
        crate::feedback::record_tool_feedback(&self.db, &input.tool, &input.rule_id, &input.entity, input.accepted).ok();
        let action = crate::feedback::get_feedback_adjustment(&self.db, &input.tool, &input.rule_id, &input.entity);
        serde_json::json!({"recorded": true, "current_action": format!("{:?}", action)}).to_string()
    }

    // --- STARK Proof Tools ---

    #[tool(name = "grav_prove_confinement", description = "Generate STARK proof that constraint masking was correctly applied. Proves: constrained[i] == raw[i] * mask[i] for all i.")]
    fn grav_prove_confinement(&self, Parameters(input): Parameters<GravProveConfinementInput>) -> String {
        if input.raw_coefficients.len() != input.mask.len() || input.mask.len() != input.constrained.len() {
            return serde_json::json!({"error": "raw_coefficients, mask, and constrained must have same length"}).to_string();
        }
        let proof = crate::crypto::stark::confinement::prove_confinement(
            &input.raw_coefficients,
            &input.mask,
            &input.constrained,
        );
        let valid = crate::crypto::stark::confinement::verify_confinement_proof(&proof);
        self.log_event("crypto", "grav_prove_confinement", &format!("dim={} valid={}", input.raw_coefficients.len(), valid));
        serde_json::json!({
            "trace_root": proof.trace_root,
            "cp_root": proof.cp_root,
            "fri_roots": proof.fri_roots,
            "fri_last_value": proof.fri_last_value.to_string(),
            "trace_length": proof.trace_length,
            "queries_count": proof.queries.len(),
            "self_verified": valid,
        }).to_string()
    }

    #[tool(name = "grav_prove_lineage", description = "Generate STARK proof that a hash chain (event lineage) was computed correctly.")]
    fn grav_prove_lineage(&self, Parameters(input): Parameters<GravProveLineageInput>) -> String {
        if input.events.is_empty() {
            return serde_json::json!({"error": "events must not be empty"}).to_string();
        }
        let proof = crate::crypto::stark::lineage::prove_lineage(&input.events);
        let valid = crate::crypto::stark::lineage::verify_lineage_proof(&proof);
        self.log_event("crypto", "grav_prove_lineage", &format!("events={} valid={}", proof.chain_length, valid));
        serde_json::json!({
            "chain_root": proof.chain_root,
            "chain_length": proof.chain_length,
            "query_count": proof.query_indices.len(),
            "self_verified": valid,
        }).to_string()
    }

    #[tool(name = "grav_audit_proof", description = "Generate STARK audit proofs (confinement + lineage) for an agent's entire run.")]
    fn grav_audit_proof(&self, Parameters(input): Parameters<GravAuditProofInput>) -> String {
        let agents = self.agents.lock().unwrap();
        let circuits = self.circuits.lock().unwrap();

        let agent = match agents.get(&input.agent_id) {
            Some(a) => a,
            None => return serde_json::json!({"error": "agent not found"}).to_string(),
        };

        let circuit = match circuits.get(&agent.circuit_id) {
            Some(c) => c,
            None => return serde_json::json!({"error": "circuit not found"}).to_string(),
        };

        // Build confinement proofs from trajectory data
        // Note: the MCP server's AgentState doesn't store step records,
        // so we report available data. Full audit_proof() is on AgentRunner.
        let mask = circuit.active_generators.clone()
            .unwrap_or_else(|| vec![true; circuit.group.algebra_dim()]);

        self.log_event("crypto", "grav_audit_proof", &format!("agent={} steps={}", input.agent_id, agent.step_count));
        serde_json::json!({
            "agent_id": input.agent_id,
            "step_count": agent.step_count,
            "circuit_id": agent.circuit_id,
            "mask": mask,
            "algebra_dim": circuit.group.algebra_dim(),
            "note": "For full STARK audit proofs, use AgentRunner.audit_proof() in the Rust API. MCP grav_prove_confinement/grav_prove_lineage tools can verify individual steps.",
        }).to_string()
    }
}

#[tool_handler]
impl ServerHandler for GravRailServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build()
        )
        .with_instructions(
            "GravRail: Deterministic geometric confinement for AI agents. \
             Every LLM output is mapped through Lie algebra → exponential map → group element, \
             guaranteeing all state transitions stay on-group. Use grav_circuit_create to define \
             a confinement space, grav_agent_create to spawn agents, and grav_step to process \
             LLM outputs through the confinement pipeline."
        )
    }
}
