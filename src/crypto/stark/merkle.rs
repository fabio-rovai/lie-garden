use super::field::FieldElement;
use sha2::{Sha256, Digest};

fn sha256_hex(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[derive(Clone)]
pub struct MerkleTree {
    pub data: Vec<FieldElement>,
    pub facts: std::collections::HashMap<String, (String, String)>,
    pub root: String,
}

impl MerkleTree {
    pub fn new(data: Vec<FieldElement>) -> Self {
        assert!(!data.is_empty());
        let num_leaves = (data.len() as f64).log2().ceil().exp2() as usize;
        let mut padded_data = data;
        padded_data.resize(num_leaves, FieldElement::zero());
        MerkleTree {
            data: padded_data,
            facts: std::collections::HashMap::new(),
            root: String::new(),
        }
    }

    pub fn get_authentication_path(&self, leaf_id: usize) -> Vec<String> {
        assert!(leaf_id < self.data.len());
        let node_id = leaf_id + self.data.len();
        let mut cur = self.root.clone();
        let mut decommitment = Vec::new();

        for bit in format!("{:b}", node_id).chars().skip(1) {
            let res = self.facts.get(&cur).unwrap();
            cur = res.0.clone();
            let mut auth = res.1.clone();
            if bit == '1' {
                (auth, cur) = (cur, auth);
            }
            decommitment.push(auth);
        }
        decommitment
    }

    fn recursive_build_tree(&mut self, node_id: usize) -> String {
        if node_id >= self.data.len() {
            let id_in_data = node_id - self.data.len();
            let leaf_data = self.data[id_in_data].to_string();
            let h = sha256_hex(&leaf_data);
            self.facts.insert(h.clone(), (leaf_data, String::new()));
            h
        } else {
            let left = self.recursive_build_tree(node_id * 2);
            let right = self.recursive_build_tree(node_id * 2 + 1);
            let h = sha256_hex(&format!("{}{}", left, right));
            self.facts.insert(h.clone(), (left, right));
            h
        }
    }

    pub fn build_tree(&mut self) {
        self.root = self.recursive_build_tree(1);
    }
}

pub fn verify_decommitment(
    leaf_id: usize,
    leaf_data: &FieldElement,
    decommitment: &[String],
    root: &str,
) -> bool {
    let leaf_num = 2_usize.pow(decommitment.len() as u32);
    let node_id = leaf_id + leaf_num;

    let mut cur = sha256_hex(&leaf_data.to_string());
    let bits: Vec<char> = format!("{:b}", node_id).chars().rev().collect();

    for (bit, auth) in bits.into_iter().zip(decommitment.iter().rev()) {
        let h = if bit == '0' {
            format!("{}{}", cur, auth)
        } else {
            format!("{}{}", auth, cur)
        };
        cur = sha256_hex(&h);
    }

    cur == root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merkle_roundtrip() {
        let data: Vec<FieldElement> = (0..8).map(|i| FieldElement::from(i as i128)).collect();
        let mut tree = MerkleTree::new(data.clone());
        tree.build_tree();

        for leaf_id in 0..8 {
            let path = tree.get_authentication_path(leaf_id);
            assert!(verify_decommitment(
                leaf_id,
                &data[leaf_id],
                &path,
                &tree.root
            ));
        }
    }

    #[test]
    fn test_tampered_leaf_fails() {
        let data: Vec<FieldElement> = (0..8).map(|i| FieldElement::from(i as i128)).collect();
        let mut tree = MerkleTree::new(data.clone());
        tree.build_tree();

        let path = tree.get_authentication_path(3);
        let tampered = FieldElement::from(999i128);
        assert!(!verify_decommitment(3, &tampered, &path, &tree.root));
    }
}
