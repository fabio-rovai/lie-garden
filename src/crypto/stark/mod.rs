//! STARK proof primitives vendored from stark-101-rs.
//!
//! Finite field arithmetic over F_p where p = 3 * 2^30 + 1 (Solinas prime),
//! polynomial operations, Merkle commitments, and Fiat-Shamir channel.
//! Used for cryptographically sound proofs replacing broken continuous-group crypto.

pub mod field;
pub mod polynomial;
pub mod merkle;
pub mod channel;
pub mod confinement;
pub mod lineage;
mod list_utils;

pub use field::FieldElement;
pub use polynomial::Polynomial;
pub use merkle::{MerkleTree, verify_decommitment};
pub use channel::Channel;
pub use confinement::{prove_confinement, verify_confinement_proof, ConfinementProof};
pub use lineage::{prove_lineage, verify_lineage_proof, LineageProof};
