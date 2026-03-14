use gravrail::gauge::bundle::FiberBundle;
use gravrail::gauge::connection::Connection;
use gravrail::lie::group::{LieGroup, GroupType};
use proptest::prelude::*;

#[test]
fn flat_connection_trivial_holonomy() {
    let group = LieGroup::new(GroupType::SO, 3);
    let bundle = FiberBundle::new(group.clone(), 2);
    let conn = Connection::flat(&bundle);

    let path = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![-1.0, 0.0],
        vec![0.0, -1.0],
    ];

    let holonomy = conn.holonomy(&path);
    let id = group.identity();
    for (a, b) in holonomy.matrix().iter().zip(id.matrix().iter()) {
        assert!((a - b).abs() < 1e-8, "Flat connection must have trivial holonomy");
    }
}

proptest! {
    #[test]
    fn gauge_transform_preserves_holonomy(
        theta in -3.14f64..3.14,
    ) {
        let group = LieGroup::new(GroupType::SO, 3);
        let bundle = FiberBundle::new(group.clone(), 2);
        let conn = Connection::with_curvature(&bundle, theta);

        let path = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];

        let h1 = conn.holonomy(&path);

        // Apply gauge transform
        let gauge_coeffs = vec![0.3, -0.2, 0.1];
        let gauge_elem = group.exp(&gauge_coeffs);
        let conn2 = conn.gauge_transform(&gauge_elem);
        let h2 = conn2.holonomy(&path);

        // Holonomy conjugates: h2 = g * h1 * g^{-1}
        let g_inv = group.inverse(&gauge_elem);
        let expected = group.multiply(&gauge_elem, &group.multiply(&h1, &g_inv));
        for (a, b) in h2.matrix().iter().zip(expected.matrix().iter()) {
            prop_assert!((a - b).abs() < 1e-6, "Gauge transform must conjugate holonomy");
        }
    }
}
