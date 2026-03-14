use crate::lie::algebra::LieAlgebra;
use crate::lie::group::LieGroup;

/// Killing form B(X,Y) = tr(ad(X) * ad(Y))
pub fn killing_form(alg: &LieAlgebra, x: &[f64], y: &[f64]) -> f64 {
    let dim = alg.dim();
    let mut trace = 0.0;
    for k in 0..dim {
        let ek = alg.basis_element(k);
        let ad_x_ek = alg.bracket(x, &ek);
        let ad_y_ad_x_ek = alg.bracket(y, &ad_x_ek);
        if k < ad_y_ad_x_ek.len() {
            trace += ad_y_ad_x_ek[k];
        }
    }
    trace
}

/// Killing form matrix B_{ij} = B(E_i, E_j)
pub fn killing_matrix(alg: &LieAlgebra) -> Vec<Vec<f64>> {
    let dim = alg.dim();
    let mut matrix = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            let ei = alg.basis_element(i);
            let ej = alg.basis_element(j);
            matrix[i][j] = killing_form(alg, &ei, &ej);
        }
    }
    matrix
}

/// Check if algebra is semisimple (Killing form is non-degenerate).
pub fn is_semisimple(alg: &LieAlgebra) -> bool {
    let km = killing_matrix(alg);
    let n = km.len();
    let m = nalgebra::DMatrix::from_fn(n, n, |i, j| km[i][j]);
    m.determinant().abs() > 1e-10
}

/// Casimir operator: C = sum g^{ij} E_i E_j where g^{ij} is inverse Killing form.
pub fn casimir_operator(alg: &LieAlgebra) -> Option<nalgebra::DMatrix<f64>> {
    let km = killing_matrix(alg);
    let n = km.len();
    let killing_m = nalgebra::DMatrix::from_fn(n, n, |i, j| km[i][j]);
    let killing_inv = killing_m.try_inverse()?;

    let group = LieGroup::new(alg.group_type, alg.n);
    let dim = group.matrix_dim();
    let mut casimir = nalgebra::DMatrix::zeros(dim, dim);

    for i in 0..n {
        for j in 0..n {
            let ei = alg.basis_element(i);
            let ej = alg.basis_element(j);
            let ei_mat = group.coefficients_to_algebra_matrix(&ei);
            let ej_mat = group.coefficients_to_algebra_matrix(&ej);
            casimir += &ei_mat * &ej_mat * killing_inv[(i, j)];
        }
    }

    Some(casimir)
}
