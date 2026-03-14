use crate::gauge::bundle::FiberBundle;
use crate::lie::group::GroupElement;

/// A connection on a principal bundle — encodes how fibers relate.
#[derive(Debug, Clone)]
pub struct Connection {
    bundle: FiberBundle,
    /// Connection coefficients: for each base direction, an algebra element.
    gauge_field: Vec<Vec<f64>>,
}

impl Connection {
    /// Flat connection — zero gauge field.
    pub fn flat(bundle: &FiberBundle) -> Self {
        let alg_dim = bundle.group.algebra_dim();
        Self {
            bundle: bundle.clone(),
            gauge_field: vec![vec![0.0; alg_dim]; bundle.base_dim],
        }
    }

    /// Connection with non-trivial curvature (for testing).
    pub fn with_curvature(bundle: &FiberBundle, strength: f64) -> Self {
        let alg_dim = bundle.group.algebra_dim();
        let mut gauge_field = vec![vec![0.0; alg_dim]; bundle.base_dim];
        if gauge_field.len() >= 2 && alg_dim >= 2 {
            gauge_field[0][0] = strength;
            gauge_field[1][1] = strength;
        } else if !gauge_field.is_empty() && alg_dim > 0 {
            gauge_field[0][0] = strength;
        }
        Self {
            bundle: bundle.clone(),
            gauge_field,
        }
    }

    /// Parallel transport along a path segment.
    pub fn transport_segment(&self, direction: &[f64]) -> GroupElement {
        let alg_dim = self.bundle.group.algebra_dim();
        let mut a_coeffs = vec![0.0; alg_dim];
        for (mu, &d) in direction.iter().enumerate() {
            if mu < self.gauge_field.len() {
                for (i, coeff) in a_coeffs.iter_mut().enumerate().take(alg_dim) {
                    *coeff += d * self.gauge_field[mu][i];
                }
            }
        }
        let neg: Vec<f64> = a_coeffs.iter().map(|x| -x).collect();
        self.bundle.group.exp(&neg)
    }

    /// Holonomy around a closed path.
    pub fn holonomy(&self, path: &[Vec<f64>]) -> GroupElement {
        let mut result = self.bundle.group.identity();
        for direction in path {
            let segment = self.transport_segment(direction);
            result = self.bundle.group.multiply(&result, &segment);
        }
        result
    }

    /// Apply gauge transformation: A'_mu = Ad(g)(A_mu) for constant gauge transform.
    pub fn gauge_transform(&self, g: &GroupElement) -> Connection {
        let repr = crate::lie::represent::Representation::new(self.bundle.group.clone(), self.bundle.group.matrix_dim());
        let new_field: Vec<Vec<f64>> = self.gauge_field.iter().map(|a_mu| {
            repr.adjoint(g, a_mu)
        }).collect();
        Connection {
            bundle: self.bundle.clone(),
            gauge_field: new_field,
        }
    }

    /// Access gauge field.
    pub fn gauge_field(&self) -> &[Vec<f64>] {
        &self.gauge_field
    }

    /// Access bundle.
    pub fn bundle(&self) -> &FiberBundle {
        &self.bundle
    }
}
