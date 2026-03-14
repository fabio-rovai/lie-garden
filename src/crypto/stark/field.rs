use std::fmt;
use rand::Rng;

/// Goldilocks prime: p = 2^64 - 2^32 + 1 = 18446744069414584321
/// 64-bit NTT-friendly prime. p-1 has factor 2^32, enabling large power-of-two
/// subgroups for FRI/NTT. Security: ~64-bit field, suitable for STARK proofs.
const K_MODULUS: u64 = 0xFFFF_FFFF_0000_0001; // 2^64 - 2^32 + 1

/// Primitive root of the Goldilocks field.
/// 7 is a generator of the multiplicative group of order p-1.
const GENERATOR_VAL: u64 = 7;

#[derive(Clone, Copy)]
pub struct FieldElement {
    pub val: u64,
}

/// A trait to represent zero of a numeric type.
pub trait Zero {
    fn zero() -> Self;
}

/// Reduce a u128 value mod p (Goldilocks prime).
/// p = 2^64 - 2^32 + 1, so 2^64 ≡ 2^32 - 1 (mod p).
#[inline(always)]
fn goldilocks_reduce(x: u128) -> u64 {
    (x % K_MODULUS as u128) as u64
}

impl FieldElement {
    pub fn new(val: i128) -> Self {
        let modulus = K_MODULUS as i128;
        let normalized = ((val % modulus) + modulus) % modulus;
        FieldElement { val: normalized as u64 }
    }

    pub fn zero() -> FieldElement {
        FieldElement { val: 0 }
    }

    pub fn one() -> FieldElement {
        FieldElement { val: 1 }
    }

    pub fn generator() -> FieldElement {
        FieldElement { val: GENERATOR_VAL }
    }

    pub fn modulus() -> i128 {
        K_MODULUS as i128
    }

    pub fn random_element() -> FieldElement {
        let mut rng = rand::thread_rng();
        let random_integer = rng.gen_range(1..K_MODULUS);
        FieldElement { val: random_integer }
    }

    /// Multiplicative inverse via extended Euclidean algorithm.
    pub fn inverse(self) -> FieldElement {
        let (mut t, mut new_t): (i128, i128) = (0, 1);
        let (mut r, mut new_r): (i128, i128) = (K_MODULUS as i128, self.val as i128);

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        let modulus = K_MODULUS as i128;
        let result = ((t % modulus) + modulus) % modulus;
        FieldElement { val: result as u64 }
    }

    /// Binary exponentiation.
    pub fn pow(&self, mut exponent: i128) -> FieldElement {
        if exponent < 0 {
            return self.inverse().pow(-exponent);
        }
        let mut cur_pow = *self;
        let mut res = FieldElement::one();

        while exponent > 0 {
            if exponent & 1 != 0 {
                res = res * cur_pow;
            }
            exponent >>= 1;
            if exponent > 0 {
                cur_pow = cur_pow * cur_pow;
            }
        }
        res
    }

    /// Checks if element has exactly the specified multiplicative order.
    pub fn is_order(&self, n: i128) -> bool {
        let mut h = FieldElement::one();
        for _ in 1..n {
            h = h * *self;
            if h == FieldElement::one() {
                return false;
            }
        }
        h * *self == FieldElement::one()
    }
}

impl fmt::Display for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl PartialEq for FieldElement {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl std::ops::Neg for FieldElement {
    type Output = FieldElement;
    fn neg(self) -> FieldElement {
        if self.val == 0 {
            self
        } else {
            FieldElement { val: K_MODULUS - self.val }
        }
    }
}

impl std::ops::Add for FieldElement {
    type Output = FieldElement;
    fn add(self, other: FieldElement) -> FieldElement {
        let sum = self.val as u128 + other.val as u128;
        let val = if sum >= K_MODULUS as u128 {
            (sum - K_MODULUS as u128) as u64
        } else {
            sum as u64
        };
        FieldElement { val }
    }
}

impl std::ops::Sub for FieldElement {
    type Output = FieldElement;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.val >= rhs.val {
            FieldElement { val: self.val - rhs.val }
        } else {
            FieldElement { val: K_MODULUS - (rhs.val - self.val) }
        }
    }
}

impl std::ops::Mul for FieldElement {
    type Output = FieldElement;
    fn mul(self, rhs: Self) -> Self::Output {
        let product = self.val as u128 * rhs.val as u128;
        FieldElement { val: goldilocks_reduce(product) }
    }
}

impl std::ops::Div for FieldElement {
    type Output = FieldElement;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let half = K_MODULUS / 2;
        if self.val > half {
            write!(f, "-{}", K_MODULUS - self.val)
        } else {
            write!(f, "{}", self.val)
        }
    }
}

impl Zero for FieldElement {
    fn zero() -> FieldElement {
        Self::zero()
    }
}

impl From<i128> for FieldElement {
    fn from(value: i128) -> Self {
        FieldElement::new(value)
    }
}

impl From<u128> for FieldElement {
    fn from(value: u128) -> Self {
        let reduced = goldilocks_reduce(value);
        FieldElement { val: reduced }
    }
}

impl Default for FieldElement {
    fn default() -> Self {
        FieldElement { val: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_basics() {
        assert_eq!(FieldElement::zero(), FieldElement::new(0));
        assert_eq!(FieldElement::one(), FieldElement::new(1));
        assert_eq!(FieldElement::generator().val, GENERATOR_VAL);
    }

    #[test]
    fn test_inverse() {
        let r = FieldElement::random_element();
        let product = r * r.inverse();
        assert_eq!(product, FieldElement::one());
    }

    #[test]
    fn test_pow_fermat() {
        // a^(p-1) == 1 for any nonzero a (Fermat's little theorem)
        let g = FieldElement::generator();
        assert_eq!(g.pow(K_MODULUS as i128 - 1), FieldElement::one());
    }

    #[test]
    fn test_is_order() {
        // p-1 = 2^64 - 2^32 = 2^32 * (2^32 - 1)
        // g^((p-1)/1024) has order 1024
        let p_minus_1 = K_MODULUS as i128 - 1;
        let sub_gen = FieldElement::generator().pow(p_minus_1 / 1024);
        assert!(sub_gen.is_order(1024));
    }

    #[test]
    fn test_goldilocks_reduction() {
        // (2^32)^2 = 2^64 ≡ 2^32 - 1 (mod p)
        let a = FieldElement::new(1i128 << 32);
        let b = FieldElement::new(1i128 << 32);
        let c = a * b;
        assert_eq!(c, FieldElement::new((1i128 << 32) - 1));
    }

    #[test]
    fn test_negative_values() {
        let a = FieldElement::new(-1);
        assert_eq!(a.val, K_MODULUS - 1);
        let b = FieldElement::new(-5);
        assert_eq!(b.val, K_MODULUS - 5);
    }

    #[test]
    fn test_add_sub_identity() {
        let a = FieldElement::random_element();
        let b = FieldElement::random_element();
        assert_eq!(a + b - b, a);
        assert_eq!(a - a, FieldElement::zero());
    }

    #[test]
    fn test_mul_distributive() {
        let a = FieldElement::random_element();
        let b = FieldElement::random_element();
        let c = FieldElement::random_element();
        assert_eq!(a * (b + c), a * b + a * c);
    }
}
