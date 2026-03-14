use crate::lie::group::{LieGroup, GroupElement};

/// Check if an observable is gauge-invariant.
/// Returns the absolute difference |O(state) - O(g * state * g^{-1})|.
pub fn check_gauge_invariance<F>(
    group: &LieGroup,
    state: &GroupElement,
    gauge_element: &GroupElement,
    observable: F,
) -> f64
where
    F: Fn(&GroupElement) -> f64,
{
    let transformed = group.multiply(gauge_element, &group.multiply(state, &group.inverse(gauge_element)));
    let o_original = observable(state);
    let o_transformed = observable(&transformed);
    (o_original - o_transformed).abs()
}
