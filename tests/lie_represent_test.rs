use gravrail::lie::group::{LieGroup, GroupType, GroupElement};
use gravrail::lie::represent::Representation;
use proptest::prelude::*;

fn arb_so3_element() -> impl Strategy<Value = GroupElement> {
    prop::array::uniform3(-1.0f64..1.0f64).prop_map(|coeffs| {
        let g = LieGroup::new(GroupType::SO, 3);
        g.exp(&coeffs.to_vec())
    })
}

proptest! {
    #[test]
    fn fundamental_homomorphism(g_elem in arb_so3_element(), h_elem in arb_so3_element()) {
        let group = LieGroup::new(GroupType::SO, 3);
        let repr = Representation::new(group.clone(), 3);
        let err = repr.verify_homomorphism(&g_elem, &h_elem);
        prop_assert!(err < 1e-8, "Representation must be a homomorphism, error: {}", err);
    }
}
