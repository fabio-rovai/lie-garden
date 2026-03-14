use schemars::JsonSchema;
use serde::Deserialize;

// === Lie Group Tools ===

#[derive(Deserialize, JsonSchema)]
pub struct GravExpInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
    /// Lie algebra coefficients to exponentiate
    pub coefficients: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravLogInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
    /// Flat matrix data of the group element (row-major, n*n entries)
    pub matrix: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravMultiplyInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
    /// First group element matrix data
    pub a: Vec<f64>,
    /// Second group element matrix data
    pub b: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravInverseInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
    /// Group element matrix data
    pub matrix: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravIdentityInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravIsMemberInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
    /// Group element matrix data to check
    pub matrix: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravProjectInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the group
    pub n: usize,
    /// Matrix data to project onto the group
    pub matrix: Vec<f64>,
}

// === Lie Algebra Tools ===

#[derive(Deserialize, JsonSchema)]
pub struct GravBracketInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// First algebra element coefficients
    pub x: Vec<f64>,
    /// Second algebra element coefficients
    pub y: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravKillingFormInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// First algebra element coefficients
    pub x: Vec<f64>,
    /// Second algebra element coefficients
    pub y: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravAlgebraInfoInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
}

// === Gauge Tools ===

#[derive(Deserialize, JsonSchema)]
pub struct GravBundleCreateInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n of the structure group
    pub n: usize,
    /// Dimension of the base manifold
    pub base_dim: usize,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravHolonomyInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Base manifold dimension
    pub base_dim: usize,
    /// Curvature strength (0.0 for flat)
    pub curvature: f64,
    /// Path as list of base-space points (each point is a vector)
    pub path: Vec<Vec<f64>>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravGaugeInvarianceInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// State matrix data
    pub state: Vec<f64>,
    /// Gauge element algebra coefficients (will be exponentiated)
    pub gauge_coefficients: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravCurvatureInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Base manifold dimension
    pub base_dim: usize,
    /// Curvature strength
    pub curvature: f64,
}

// === Crypto Tools ===

#[derive(Deserialize, JsonSchema)]
pub struct GravCommitInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Value to commit (algebra coefficients)
    pub value: Vec<f64>,
    /// Randomness (algebra coefficients)
    pub randomness: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravVerifyCommitInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Commitment matrix data
    pub commitment: Vec<f64>,
    /// Claimed value
    pub value: Vec<f64>,
    /// Claimed randomness
    pub randomness: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravSignInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Message to sign (UTF-8 string)
    pub message: String,
    /// Secret key (algebra coefficients)
    pub secret: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravVerifySignInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Message that was signed
    pub message: String,
    /// Public key matrix data
    pub public_key: Vec<f64>,
    /// Signature R matrix data
    pub sig_r: Vec<f64>,
    /// Signature s algebra coefficients
    pub sig_s: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravDiffieHellmanInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Alice's secret (algebra coefficients)
    pub secret_a: Vec<f64>,
    /// Bob's secret (algebra coefficients)
    pub secret_b: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravZkProveInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Secret witness (algebra coefficients)
    pub secret: Vec<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravZkVerifyInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Public value matrix data
    pub public_key: Vec<f64>,
    /// Proof commitment matrix data
    pub proof_commitment: Vec<f64>,
    /// Proof response algebra coefficients
    pub proof_response: Vec<f64>,
}

// === Circuit Tools ===

#[derive(Deserialize, JsonSchema)]
pub struct GravCircuitCreateInput {
    /// Group type: SO, SE, or GL
    pub group_type: String,
    /// Dimension n
    pub n: usize,
    /// Optional: which generators are active (bitmask)
    pub active_generators: Option<Vec<bool>>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravCircuitStepInput {
    /// Circuit ID
    pub circuit_id: String,
    /// Agent ID
    pub agent_id: String,
    /// LLM output text to confine
    pub text: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravCircuitAgentCreateInput {
    /// Circuit ID to spawn agent in
    pub circuit_id: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravCircuitAgentStateInput {
    /// Circuit ID
    pub circuit_id: String,
    /// Agent ID
    pub agent_id: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravCircuitLockInput {
    /// Circuit ID to lock
    pub circuit_id: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravCircuitListInput {
    /// Optional filter by group type
    pub group_type: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravMapToAlgebraInput {
    /// Text to map
    pub text: String,
    /// Algebra dimension
    pub algebra_dim: usize,
    /// Scale factor (default 1.0)
    pub scale: Option<f64>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravReachabilityInput {
    /// Circuit ID
    pub circuit_id: String,
    /// Sample algebra inputs to explore
    pub sample_inputs: Vec<Vec<f64>>,
    /// Maximum depth (steps)
    pub max_depth: usize,
    /// Maximum breadth (states per level)
    pub max_breadth: usize,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravTrajectoryInput {
    /// Circuit ID
    pub circuit_id: String,
    /// Agent ID
    pub agent_id: String,
}

// === Lifecycle Tools ===

#[derive(Deserialize, JsonSchema)]
pub struct GravLineageInput {
    /// Optional session ID (defaults to current session)
    pub session_id: Option<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravLineageVerifyInput {
    /// Session ID to verify
    pub session_id: String,
}

#[derive(Deserialize, JsonSchema)]
pub struct GravFeedbackInput {
    /// Tool name
    pub tool: String,
    /// Rule ID
    pub rule_id: String,
    /// Entity (e.g. circuit ID)
    pub entity: String,
    /// Whether the action was accepted
    pub accepted: bool,
}
