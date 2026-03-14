use gravrail::lie::algebra::LieAlgebra;
use gravrail::lie::group::GroupType;
use proptest::prelude::*;

proptest! {
    #[test]
    fn bracket_antisymmetry(
        a in prop::collection::vec(-1.0f64..1.0, 3),
        b in prop::collection::vec(-1.0f64..1.0, 3),
    ) {
        let alg = LieAlgebra::new(GroupType::SO, 3);
        let ab = alg.bracket(&a, &b);
        let ba = alg.bracket(&b, &a);
        // [X,Y] = -[Y,X]
        for (x, y) in ab.iter().zip(ba.iter()) {
            prop_assert!((x + y).abs() < 1e-10);
        }
    }

    #[test]
    fn jacobi_identity(
        a in prop::collection::vec(-1.0f64..1.0, 3),
        b in prop::collection::vec(-1.0f64..1.0, 3),
        c in prop::collection::vec(-1.0f64..1.0, 3),
    ) {
        let alg = LieAlgebra::new(GroupType::SO, 3);
        // [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
        let bc = alg.bracket(&b, &c);
        let ca = alg.bracket(&c, &a);
        let ab = alg.bracket(&a, &b);
        let t1 = alg.bracket(&a, &bc);
        let t2 = alg.bracket(&b, &ca);
        let t3 = alg.bracket(&c, &ab);
        for i in 0..t1.len() {
            prop_assert!((t1[i] + t2[i] + t3[i]).abs() < 1e-8, "Jacobi identity violated");
        }
    }
}
