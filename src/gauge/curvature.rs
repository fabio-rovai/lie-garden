use crate::gauge::connection::Connection;
use crate::lie::algebra::LieAlgebra;

/// Field strength tensor F_{mu,nu} = [A_mu, A_nu] (for constant connections).
pub fn field_strength(conn: &Connection, mu: usize, nu: usize) -> Vec<f64> {
    let field = conn.gauge_field();
    let group = &conn.bundle().group;
    let alg = LieAlgebra::new(group.group_type, group.n);

    if mu >= field.len() || nu >= field.len() {
        return vec![0.0; alg.dim()];
    }

    alg.bracket(&field[mu], &field[nu])
}

/// Check if connection is flat (F = 0 everywhere).
pub fn is_flat(conn: &Connection) -> bool {
    let field = conn.gauge_field();
    let group = &conn.bundle().group;
    let alg = LieAlgebra::new(group.group_type, group.n);

    for mu in 0..field.len() {
        for nu in (mu + 1)..field.len() {
            let f = alg.bracket(&field[mu], &field[nu]);
            if f.iter().any(|x| x.abs() > 1e-10) {
                return false;
            }
        }
    }
    true
}
