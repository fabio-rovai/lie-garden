use crate::lie::group::LieGroup;
use crate::util;

/// A circuit defines the confinement space — a Lie group + optional algebra constraints.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub group: LieGroup,
    /// Optional: restrict which algebra generators are active (bitmask).
    /// None = all generators active.
    pub active_generators: Option<Vec<bool>>,
    pub id: String,
    pub locked: bool,
}

impl Circuit {
    pub fn new(group: LieGroup, active_generators: Option<Vec<bool>>) -> Self {
        Self {
            group,
            active_generators,
            id: util::rand_id(),
            locked: false,
        }
    }

    /// Constrain algebra element to only active generators.
    pub fn constrain_algebra(&self, coefficients: &[f64]) -> Vec<f64> {
        match &self.active_generators {
            Some(mask) => {
                coefficients.iter().enumerate().map(|(i, &c)| {
                    if i < mask.len() && mask[i] { c } else { 0.0 }
                }).collect()
            }
            None => coefficients.to_vec(),
        }
    }

    pub fn lock(&mut self) {
        self.locked = true;
    }
}
