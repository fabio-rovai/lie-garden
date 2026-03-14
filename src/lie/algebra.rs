use crate::lie::group::{GroupType, LieGroup};

pub struct LieAlgebra {
    pub group_type: GroupType,
    pub n: usize,
    group: LieGroup,
}

impl LieAlgebra {
    pub fn new(group_type: GroupType, n: usize) -> Self {
        Self { group_type, n, group: LieGroup::new(group_type, n) }
    }

    /// Lie bracket [X, Y] = XY - YX (as algebra coefficient vectors).
    pub fn bracket(&self, x_coeffs: &[f64], y_coeffs: &[f64]) -> Vec<f64> {
        let x_mat = self.group.coefficients_to_algebra_matrix(x_coeffs);
        let y_mat = self.group.coefficients_to_algebra_matrix(y_coeffs);
        let bracket = &x_mat * &y_mat - &y_mat * &x_mat;
        self.group.algebra_matrix_to_coefficients(&bracket)
    }

    /// Structure constants c^k_{ij}: [E_i, E_j] = sum c^k_{ij} E_k
    pub fn structure_constants(&self) -> Vec<Vec<Vec<f64>>> {
        let dim = self.group.algebra_dim();
        let mut constants = vec![vec![vec![0.0; dim]; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                let mut ei = vec![0.0; dim];
                let mut ej = vec![0.0; dim];
                ei[i] = 1.0;
                ej[j] = 1.0;
                let bracket = self.bracket(&ei, &ej);
                for k in 0..dim {
                    constants[i][j][k] = bracket[k];
                }
            }
        }
        constants
    }

    /// Algebra dimension.
    pub fn dim(&self) -> usize {
        self.group.algebra_dim()
    }

    /// Get basis element (unit vector in coefficient space).
    pub fn basis_element(&self, index: usize) -> Vec<f64> {
        let mut e = vec![0.0; self.dim()];
        if index < e.len() {
            e[index] = 1.0;
        }
        e
    }
}
