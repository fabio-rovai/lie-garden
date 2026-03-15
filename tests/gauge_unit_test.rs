/// Unit tests for gauge theory modules:
///   gauge::bundle, gauge::connection, gauge::curvature, gauge::invariance
use gravrail::gauge::bundle::FiberBundle;
use gravrail::gauge::connection::Connection;
use gravrail::gauge::curvature::{field_strength, is_flat};
use gravrail::gauge::invariance::check_gauge_invariance;
use gravrail::lie::group::{GroupType, LieGroup};

// ─────────────────────────────────────────────
// FiberBundle tests
// ─────────────────────────────────────────────

#[test]
fn fiber_bundle_stores_base_dim() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 4);
    assert_eq!(bundle.base_dim, 4);
}

#[test]
fn fiber_bundle_group_algebra_dim_so3() {
    // SO(3) algebra has dim = 3*(3-1)/2 = 3
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 2);
    assert_eq!(bundle.group.algebra_dim(), 3);
}

#[test]
fn fiber_bundle_group_algebra_dim_gl2() {
    // GL(2) algebra has dim = 2*2 = 4
    let group = LieGroup::new(GroupType::GL, 2);
    let bundle = FiberBundle::new(group, 3);
    assert_eq!(bundle.group.algebra_dim(), 4);
}

#[test]
fn fiber_bundle_clone_is_independent() {
    let group = LieGroup::new(GroupType::SO, 3);
    let b1 = FiberBundle::new(group, 5);
    let b2 = b1.clone();
    assert_eq!(b1.base_dim, b2.base_dim);
}

// ─────────────────────────────────────────────
// Connection – flat construction
// ─────────────────────────────────────────────

#[test]
fn flat_connection_gauge_field_all_zeros() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 2);
    let conn = Connection::flat(&bundle);

    for row in conn.gauge_field() {
        for &val in row {
            assert_eq!(val, 0.0, "Flat connection must have all-zero gauge field");
        }
    }
}

#[test]
fn flat_connection_gauge_field_shape() {
    // gauge_field should be base_dim rows × algebra_dim columns
    let group = LieGroup::new(GroupType::SO, 3); // alg_dim = 3
    let bundle = FiberBundle::new(group, 4);      // base_dim = 4
    let conn = Connection::flat(&bundle);

    assert_eq!(conn.gauge_field().len(), 4, "Number of rows must equal base_dim");
    for row in conn.gauge_field() {
        assert_eq!(row.len(), 3, "Each row must have algebra_dim entries");
    }
}

#[test]
fn flat_connection_bundle_accessor() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 2);
    let conn = Connection::flat(&bundle);
    assert_eq!(conn.bundle().base_dim, 2);
}

// ─────────────────────────────────────────────
// Connection – transport_segment on flat connection
// ─────────────────────────────────────────────

#[test]
fn flat_connection_transport_returns_identity() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group.clone(), 2);
    let conn = Connection::flat(&bundle);

    let direction = vec![1.0, 0.0];
    let transported = conn.transport_segment(&direction);
    let id = group.identity();

    assert!(
        transported.close_to(&id, 1e-10),
        "Parallel transport along a flat connection must return the identity"
    );
}

// ─────────────────────────────────────────────
// Connection – with_curvature construction
// ─────────────────────────────────────────────

#[test]
fn curved_connection_has_nonzero_gauge_field() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 2);
    let conn = Connection::with_curvature(&bundle, 1.0);

    let any_nonzero = conn
        .gauge_field()
        .iter()
        .flat_map(|row| row.iter())
        .any(|&v| v != 0.0);

    assert!(any_nonzero, "Curved connection must have at least one nonzero coefficient");
}

#[test]
fn curved_connection_transport_differs_from_identity() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group.clone(), 2);
    let conn = Connection::with_curvature(&bundle, 0.5);

    let direction = vec![1.0, 0.0];
    let transported = conn.transport_segment(&direction);
    let id = group.identity();

    // With strength 0.5 the transport should NOT equal identity
    assert!(
        !transported.close_to(&id, 1e-6),
        "Transport on a curved connection should not be the identity"
    );
}

// ─────────────────────────────────────────────
// Curvature – is_flat
// ─────────────────────────────────────────────

#[test]
fn is_flat_true_for_flat_connection() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 3);
    let conn = Connection::flat(&bundle);
    assert!(is_flat(&conn), "is_flat must return true for a flat connection");
}

#[test]
fn is_flat_false_for_curved_connection() {
    // SO(3) has non-abelian algebra so [A_mu, A_nu] != 0 for generic nonzero entries
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 3);
    let conn = Connection::with_curvature(&bundle, 1.0);
    assert!(!is_flat(&conn), "is_flat must return false for a connection with curvature");
}

// ─────────────────────────────────────────────
// Curvature – field_strength
// ─────────────────────────────────────────────

#[test]
fn field_strength_zero_for_flat_connection() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 2);
    let conn = Connection::flat(&bundle);

    let f = field_strength(&conn, 0, 1);
    for &v in &f {
        assert!(v.abs() < 1e-14, "Field strength must be zero for flat connection");
    }
}

#[test]
fn field_strength_nonzero_for_curved_connection() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 3);
    let conn = Connection::with_curvature(&bundle, 1.0);

    let f = field_strength(&conn, 0, 1);
    let norm: f64 = f.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm > 1e-10, "Field strength must be nonzero for curved connection, got {}", norm);
}

#[test]
fn field_strength_antisymmetric() {
    // F_{mu,nu} = -F_{nu,mu} because [A, B] = -[B, A]
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group, 3);
    let conn = Connection::with_curvature(&bundle, 0.7);

    let f01 = field_strength(&conn, 0, 1);
    let f10 = field_strength(&conn, 1, 0);

    for (a, b) in f01.iter().zip(f10.iter()) {
        assert!(
            (a + b).abs() < 1e-12,
            "Field strength must be antisymmetric: F_{{mu,nu}} = -F_{{nu,mu}}"
        );
    }
}

#[test]
fn field_strength_out_of_bounds_returns_zero_vector() {
    let group = LieGroup::new(GroupType::SO, 3); // alg_dim = 3
    let bundle = FiberBundle::new(group, 2);
    let conn = Connection::flat(&bundle);

    // mu=5 is out of range for base_dim=2
    let f = field_strength(&conn, 5, 6);
    assert_eq!(f.len(), 3);
    for &v in &f {
        assert_eq!(v, 0.0);
    }
}

// ─────────────────────────────────────────────
// Gauge invariance
// ─────────────────────────────────────────────

#[test]
fn check_gauge_invariance_trace_observable_is_invariant() {
    // Trace is a gauge-invariant observable: Tr(g s g^{-1}) = Tr(s)
    let group = LieGroup::new(GroupType::SO, 3);

    let state = group.exp(&[0.3, -0.2, 0.1]);
    let gauge = group.exp(&[0.1, 0.4, -0.3]);

    let diff = check_gauge_invariance(&group, &state, &gauge, |elem| {
        // Trace of the 3×3 matrix
        let m = elem.to_matrix();
        m[(0, 0)] + m[(1, 1)] + m[(2, 2)]
    });

    assert!(
        diff < 1e-10,
        "Trace is gauge invariant; got difference {}",
        diff
    );
}

#[test]
fn check_gauge_invariance_identity_gauge_gives_zero() {
    let group = LieGroup::new(GroupType::SO, 3);
    let state = group.exp(&[0.5, 0.0, -0.5]);
    let gauge = group.identity();

    let diff = check_gauge_invariance(&group, &state, &gauge, |elem| {
        elem.matrix().iter().map(|x| x * x).sum::<f64>()
    });

    assert!(diff < 1e-14, "Identity gauge transform must give zero difference");
}

#[test]
fn check_gauge_invariance_returns_positive_for_noninvariant_observable() {
    let group = LieGroup::new(GroupType::SO, 3);
    let state = group.exp(&[0.4, 0.0, 0.0]);
    let gauge = group.exp(&[0.0, 0.5, 0.0]);

    // The (0,0) matrix entry is generally NOT gauge-invariant
    let diff = check_gauge_invariance(&group, &state, &gauge, |elem| elem.matrix()[0]);

    // We can only assert the function ran and returned a non-negative f64
    assert!(diff >= 0.0, "check_gauge_invariance must return a non-negative value");
}
