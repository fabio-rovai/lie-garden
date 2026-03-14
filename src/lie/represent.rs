use crate::lie::group::{LieGroup, GroupElement};
use nalgebra::DMatrix;

/// A representation rho: G -> GL(V).
#[derive(Debug, Clone)]
pub struct Representation {
    pub group: LieGroup,
    pub dim_v: usize,
}

impl Representation {
    pub fn new(group: LieGroup, dim_v: usize) -> Self {
        Self { group, dim_v }
    }

    /// Fundamental representation: rho(g) = g (the matrix itself).
    pub fn fundamental(&self, element: &GroupElement) -> DMatrix<f64> {
        element.to_matrix()
    }

    /// Adjoint representation: Ad(g)(X) = g X g^{-1}
    pub fn adjoint(&self, g: &GroupElement, x_coeffs: &[f64]) -> Vec<f64> {
        let g_mat = g.to_matrix();
        let g_inv = self.group.inverse(g).to_matrix();
        let x_mat = self.group.coefficients_to_algebra_matrix(x_coeffs);
        let result = &g_mat * &x_mat * &g_inv;
        self.group.algebra_matrix_to_coefficients(&result)
    }

    /// Check homomorphism: rho(g*h) = rho(g) * rho(h)
    pub fn verify_homomorphism(&self, g: &GroupElement, h: &GroupElement) -> f64 {
        let gh = self.group.multiply(g, h);
        let rho_gh = self.fundamental(&gh);
        let rho_g = self.fundamental(g);
        let rho_h = self.fundamental(h);
        let rho_g_rho_h = &rho_g * &rho_h;
        (&rho_gh - &rho_g_rho_h).norm()
    }
}
