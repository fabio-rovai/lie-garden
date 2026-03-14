use gravrail::lie::group::{LieGroup, GroupType, GroupElement};
use proptest::prelude::*;

fn arb_so3_element() -> impl Strategy<Value = GroupElement> {
    prop::array::uniform3(-1.0f64..1.0f64).prop_map(|coeffs| {
        let g = LieGroup::new(GroupType::SO, 3);
        g.exp(&coeffs.to_vec())
    })
}

fn arb_se3_element() -> impl Strategy<Value = GroupElement> {
    // SE with n=3: 3x3 matrices, algebra_dim=3 (1 rotation + 2 translation)
    prop::array::uniform3(-1.0f64..1.0f64).prop_map(|coeffs| {
        let g = LieGroup::new(GroupType::SE, 3);
        g.exp(&coeffs.to_vec())
    })
}

fn arb_gl2_element() -> impl Strategy<Value = GroupElement> {
    // GL(2): algebra_dim=4
    prop::array::uniform4(-0.5f64..0.5f64).prop_map(|coeffs| {
        let g = LieGroup::new(GroupType::GL, 2);
        g.exp(&coeffs.to_vec())
    })
}

proptest! {
    // === SO(3) tests ===

    #[test]
    fn so3_closure(a in arb_so3_element(), b in arb_so3_element()) {
        let g = LieGroup::new(GroupType::SO, 3);
        let c = g.multiply(&a, &b);
        prop_assert!(g.is_member(&c), "Product must be on-group");
    }

    #[test]
    fn so3_identity(a in arb_so3_element()) {
        let g = LieGroup::new(GroupType::SO, 3);
        let e = g.identity();
        let ae = g.multiply(&a, &e);
        let ea = g.multiply(&e, &a);
        prop_assert!(a.close_to(&ae, 1e-10));
        prop_assert!(a.close_to(&ea, 1e-10));
    }

    #[test]
    fn so3_inverse(a in arb_so3_element()) {
        let g = LieGroup::new(GroupType::SO, 3);
        let a_inv = g.inverse(&a);
        let should_be_id = g.multiply(&a, &a_inv);
        prop_assert!(should_be_id.close_to(&g.identity(), 1e-10));
    }

    #[test]
    fn so3_associativity(a in arb_so3_element(), b in arb_so3_element(), c in arb_so3_element()) {
        let g = LieGroup::new(GroupType::SO, 3);
        let ab_c = g.multiply(&g.multiply(&a, &b), &c);
        let a_bc = g.multiply(&a, &g.multiply(&b, &c));
        prop_assert!(ab_c.close_to(&a_bc, 1e-10));
    }

    #[test]
    fn so3_exp_log_roundtrip(coeffs in prop::collection::vec(-0.5f64..0.5f64, 3)) {
        let g = LieGroup::new(GroupType::SO, 3);
        let elem = g.exp(&coeffs);
        let recovered = g.log(&elem);
        let re_exp = g.exp(&recovered);
        prop_assert!(elem.close_to(&re_exp, 1e-8));
    }

    #[test]
    fn so3_exp_always_on_group(coeffs in prop::collection::vec(-10.0f64..10.0f64, 3)) {
        let g = LieGroup::new(GroupType::SO, 3);
        let elem = g.exp(&coeffs);
        prop_assert!(g.is_member(&elem), "exp() must always produce on-group element");
    }

    // === SE(3) tests (2D rigid motions, 3x3 matrices) ===

    #[test]
    fn se3_closure(a in arb_se3_element(), b in arb_se3_element()) {
        let g = LieGroup::new(GroupType::SE, 3);
        let c = g.multiply(&a, &b);
        prop_assert!(g.is_member(&c), "SE product must be on-group");
    }

    #[test]
    fn se3_identity(a in arb_se3_element()) {
        let g = LieGroup::new(GroupType::SE, 3);
        let e = g.identity();
        let ae = g.multiply(&a, &e);
        let ea = g.multiply(&e, &a);
        prop_assert!(a.close_to(&ae, 1e-10));
        prop_assert!(a.close_to(&ea, 1e-10));
    }

    #[test]
    fn se3_inverse(a in arb_se3_element()) {
        let g = LieGroup::new(GroupType::SE, 3);
        let a_inv = g.inverse(&a);
        let should_be_id = g.multiply(&a, &a_inv);
        prop_assert!(should_be_id.close_to(&g.identity(), 1e-10));
    }

    #[test]
    fn se3_associativity(a in arb_se3_element(), b in arb_se3_element(), c in arb_se3_element()) {
        let g = LieGroup::new(GroupType::SE, 3);
        let ab_c = g.multiply(&g.multiply(&a, &b), &c);
        let a_bc = g.multiply(&a, &g.multiply(&b, &c));
        prop_assert!(ab_c.close_to(&a_bc, 1e-10));
    }

    #[test]
    fn se3_exp_always_on_group(coeffs in prop::collection::vec(-10.0f64..10.0f64, 3)) {
        let g = LieGroup::new(GroupType::SE, 3);
        let elem = g.exp(&coeffs);
        prop_assert!(g.is_member(&elem), "SE exp() must always produce on-group element");
    }

    // === GL(2) tests ===

    #[test]
    fn gl2_closure(a in arb_gl2_element(), b in arb_gl2_element()) {
        let g = LieGroup::new(GroupType::GL, 2);
        let c = g.multiply(&a, &b);
        prop_assert!(g.is_member(&c), "GL product must be on-group");
    }

    #[test]
    fn gl2_identity(a in arb_gl2_element()) {
        let g = LieGroup::new(GroupType::GL, 2);
        let e = g.identity();
        let ae = g.multiply(&a, &e);
        let ea = g.multiply(&e, &a);
        prop_assert!(a.close_to(&ae, 1e-10));
        prop_assert!(a.close_to(&ea, 1e-10));
    }

    #[test]
    fn gl2_inverse(a in arb_gl2_element()) {
        let g = LieGroup::new(GroupType::GL, 2);
        let a_inv = g.inverse(&a);
        let should_be_id = g.multiply(&a, &a_inv);
        prop_assert!(should_be_id.close_to(&g.identity(), 1e-8));
    }

    #[test]
    fn gl2_associativity(a in arb_gl2_element(), b in arb_gl2_element(), c in arb_gl2_element()) {
        let g = LieGroup::new(GroupType::GL, 2);
        let ab_c = g.multiply(&g.multiply(&a, &b), &c);
        let a_bc = g.multiply(&a, &g.multiply(&b, &c));
        prop_assert!(ab_c.close_to(&a_bc, 1e-8));
    }

    #[test]
    fn gl2_exp_always_on_group(coeffs in prop::collection::vec(-2.0f64..2.0f64, 4)) {
        let g = LieGroup::new(GroupType::GL, 2);
        let elem = g.exp(&coeffs);
        prop_assert!(g.is_member(&elem), "GL exp() must always produce on-group element");
    }
}
