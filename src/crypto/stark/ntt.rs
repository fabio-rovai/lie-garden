//! Number Theoretic Transform (NTT) for O(n log n) polynomial evaluation.
//!
//! The Goldilocks prime p = 2^64 - 2^32 + 1 has p-1 = 2^32 * (2^32 - 1),
//! supporting NTT domains up to size 2^32.
//!
//! NTT is the finite-field analogue of FFT: given polynomial coefficients,
//! it evaluates the polynomial at all n-th roots of unity in O(n log n).

use super::field::FieldElement;

/// Compute a primitive n-th root of unity in the Goldilocks field.
/// Requires n to be a power of 2 and n <= 2^32.
pub fn root_of_unity(n: usize) -> FieldElement {
    assert!(n.is_power_of_two(), "NTT size must be a power of 2");
    assert!(n <= (1 << 32), "NTT size exceeds 2^32 for Goldilocks");

    // p - 1 = 2^64 - 2^32 = 2^32 * (2^32 - 1)
    // g^((p-1)/n) is a primitive n-th root of unity
    let p_minus_1 = FieldElement::modulus() - 1;
    let exponent = p_minus_1 / n as i128;
    FieldElement::generator().pow(exponent)
}

/// In-place iterative Cooley-Tukey NTT (decimation-in-time).
/// Input: coefficients of length n (must be power of 2).
/// Output: evaluations at the n-th roots of unity.
pub fn ntt(coeffs: &mut [FieldElement]) {
    let n = coeffs.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two(), "NTT size must be a power of 2");

    // Bit-reversal permutation
    bit_reverse(coeffs);

    // Butterfly stages
    let omega_n = root_of_unity(n);
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let omega = omega_n.pow((n / stage_len) as i128);

        for start in (0..n).step_by(stage_len) {
            let mut w = FieldElement::one();
            for j in 0..half {
                let u = coeffs[start + j];
                let v = coeffs[start + j + half] * w;
                coeffs[start + j] = u + v;
                coeffs[start + j + half] = u - v;
                w = w * omega;
            }
        }
        stage_len <<= 1;
    }
}

/// Inverse NTT: evaluations → coefficients.
/// Applies NTT with inverse root of unity, then divides by n.
pub fn intt(evals: &mut [FieldElement]) {
    let n = evals.len();
    if n <= 1 {
        return;
    }
    assert!(n.is_power_of_two());

    // Bit-reversal permutation
    bit_reverse(evals);

    // Butterfly stages with inverse root
    let omega_n_inv = root_of_unity(n).inverse();
    let mut stage_len = 2;
    while stage_len <= n {
        let half = stage_len / 2;
        let omega = omega_n_inv.pow((n / stage_len) as i128);

        for start in (0..n).step_by(stage_len) {
            let mut w = FieldElement::one();
            for j in 0..half {
                let u = evals[start + j];
                let v = evals[start + j + half] * w;
                evals[start + j] = u + v;
                evals[start + j + half] = u - v;
                w = w * omega;
            }
        }
        stage_len <<= 1;
    }

    // Divide by n
    let n_inv = FieldElement::new(n as i128).inverse();
    for val in evals.iter_mut() {
        *val = *val * n_inv;
    }
}

/// Evaluate a polynomial at all points in a coset: {shift * omega^i} for i in 0..n.
/// This is NTT on the coefficients scaled by powers of shift.
pub fn coset_ntt(coeffs: &[FieldElement], shift: FieldElement) -> Vec<FieldElement> {
    let n = coeffs.len();
    let mut scaled = Vec::with_capacity(n);
    let mut s = FieldElement::one();
    for c in coeffs {
        scaled.push(*c * s);
        s = s * shift;
    }
    ntt(&mut scaled);
    scaled
}

/// In-place bit-reversal permutation.
fn bit_reverse(data: &mut [FieldElement]) {
    let n = data.len();
    let log_n = n.trailing_zeros();

    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if i < j {
            data.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_root_of_unity_order() {
        // omega^n == 1
        let omega = root_of_unity(1024);
        assert_eq!(omega.pow(1024), FieldElement::one());
        // omega^512 != 1 (primitive)
        assert_ne!(omega.pow(512), FieldElement::one());
    }

    #[test]
    fn test_ntt_intt_roundtrip() {
        let coeffs_raw = vec![1i128, 2, 3, 4, 5, 6, 7, 8];
        let coeffs: Vec<FieldElement> = coeffs_raw.iter().map(|&c| FieldElement::new(c)).collect();

        let mut data = coeffs.clone();
        ntt(&mut data);
        intt(&mut data);

        for (a, b) in data.iter().zip(coeffs.iter()) {
            assert_eq!(*a, *b, "NTT→INTT roundtrip failed");
        }
    }

    #[test]
    fn test_ntt_matches_naive_eval() {
        // NTT should produce the same values as evaluating the polynomial
        // at each root of unity
        let coeffs_raw = vec![3i128, 1, 4, 1];
        let coeffs: Vec<FieldElement> = coeffs_raw.iter().map(|&c| FieldElement::new(c)).collect();

        let omega = root_of_unity(4);

        // Naive evaluation at omega^0, omega^1, omega^2, omega^3
        let mut naive = Vec::with_capacity(4);
        for i in 0..4 {
            let x = omega.pow(i as i128);
            let mut val = FieldElement::zero();
            let mut x_pow = FieldElement::one();
            for c in &coeffs {
                val = val + *c * x_pow;
                x_pow = x_pow * x;
            }
            naive.push(val);
        }

        let mut ntt_result = coeffs.clone();
        ntt(&mut ntt_result);

        for (a, b) in ntt_result.iter().zip(naive.iter()) {
            assert_eq!(*a, *b, "NTT doesn't match naive evaluation");
        }
    }

    #[test]
    fn test_ntt_size_1() {
        let mut data = vec![FieldElement::new(42)];
        ntt(&mut data);
        assert_eq!(data[0], FieldElement::new(42));
    }

    #[test]
    fn test_coset_ntt() {
        let coeffs = vec![
            FieldElement::new(1),
            FieldElement::new(2),
            FieldElement::new(3),
            FieldElement::new(4),
        ];
        let shift = FieldElement::generator();
        let result = coset_ntt(&coeffs, shift);

        // Verify by naive evaluation at shift * omega^i
        let omega = root_of_unity(4);
        for i in 0..4 {
            let x = shift * omega.pow(i as i128);
            let mut val = FieldElement::zero();
            let mut x_pow = FieldElement::one();
            for c in &coeffs {
                val = val + *c * x_pow;
                x_pow = x_pow * x;
            }
            assert_eq!(result[i], val, "Coset NTT mismatch at index {}", i);
        }
    }
}
