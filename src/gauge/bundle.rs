use crate::lie::group::LieGroup;

/// A principal fiber bundle with structure group G over a base space.
#[derive(Debug, Clone)]
pub struct FiberBundle {
    pub group: LieGroup,
    pub base_dim: usize,
}

impl FiberBundle {
    pub fn new(group: LieGroup, base_dim: usize) -> Self {
        Self { group, base_dim }
    }
}
