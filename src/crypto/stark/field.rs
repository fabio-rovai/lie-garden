use std::fmt;
use rand::Rng;

#[derive(Clone, Copy)]
pub struct FieldElement {
    pub val: i128,
}

/// A trait to represent zero of a numeric type.
pub trait Zero {
    fn zero() -> Self;
}

impl FieldElement {
    /// Prime modulus: p = 3 * 2^30 + 1 = 3221225473 (Solinas prime).
    /// This field has smooth multiplicative order p-1 = 3 * 2^30,
    /// enabling efficient NTT/FRI with large power-of-two subgroups.
    const K_MODULUS: i128 = 3 * 2i128.pow(30) + 1;
    const GENERATOR_VAL: i128 = 5;

    pub fn new(val: i128) -> Self {
        let normalized_val = ((val % Self::K_MODULUS) + Self::K_MODULUS) % Self::K_MODULUS;
        FieldElement { val: normalized_val }
    }

    pub fn zero() -> FieldElement {
        FieldElement::new(0)
    }

    pub fn one() -> FieldElement {
        FieldElement::new(1)
    }

    pub fn generator() -> FieldElement {
        FieldElement::new(Self::GENERATOR_VAL)
    }

    pub fn modulus() -> i128 {
        Self::K_MODULUS
    }

    pub fn random_element() -> FieldElement {
        let mut rng = rand::thread_rng();
        let random_integer = rng.gen_range(1..Self::K_MODULUS);
        FieldElement::new(random_integer)
    }

    /// Multiplicative inverse via extended Euclidean algorithm.
    pub fn inverse(self) -> FieldElement {
        let (mut t, mut new_t) = (0i128, 1i128);
        let (mut r, mut new_r) = (Self::K_MODULUS, self.val);

        while new_r != 0 {
            let quotient = r.div_euclid(new_r);
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }
        FieldElement::new(t)
    }

    /// Binary exponentiation.
    pub fn pow(&self, mut exponent: i128) -> FieldElement {
        let mut cur_pow = *self;
        let mut res = FieldElement::new(1);

        while exponent > 0 {
            if exponent % 2 != 0 {
                res = res * cur_pow;
            }
            exponent = exponent.div_euclid(2);
            cur_pow = cur_pow * cur_pow;
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
        let a = ((self.val % Self::K_MODULUS) + Self::K_MODULUS) % Self::K_MODULUS;
        let b = ((other.val % Self::K_MODULUS) + Self::K_MODULUS) % Self::K_MODULUS;
        a == b
    }
}

impl std::ops::Neg for FieldElement {
    type Output = FieldElement;
    fn neg(self) -> FieldElement {
        FieldElement::new(Self::K_MODULUS - self.val)
    }
}

impl std::ops::Add for FieldElement {
    type Output = FieldElement;
    fn add(self, other: FieldElement) -> FieldElement {
        FieldElement { val: (self.val + other.val) % Self::K_MODULUS }
    }
}

impl std::ops::Sub for FieldElement {
    type Output = FieldElement;
    fn sub(self, rhs: Self) -> Self::Output {
        FieldElement::new(self.val - rhs.val)
    }
}

impl std::ops::Mul for FieldElement {
    type Output = FieldElement;
    fn mul(self, rhs: Self) -> Self::Output {
        FieldElement::new(self.val * rhs.val)
    }
}

impl std::ops::Div for FieldElement {
    type Output = FieldElement;
    fn div(self, rhs: Self) -> Self::Output {
        FieldElement::new(rhs.inverse().val * self.val)
    }
}

impl fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            (self.val + Self::K_MODULUS.div_euclid(2)) % Self::K_MODULUS
                - Self::K_MODULUS.div_euclid(2)
        )
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
        let reduced = (value % Self::K_MODULUS as u128) as i128;
        FieldElement::new(reduced)
    }
}

impl Default for FieldElement {
    fn default() -> Self {
        FieldElement { val: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::FieldElement;

    #[test]
    fn test_field_basics() {
        assert_eq!(FieldElement::zero(), FieldElement::new(0));
        assert_eq!(FieldElement::one(), FieldElement::new(1));
        assert_eq!(FieldElement::generator().val, 5);
    }

    #[test]
    fn test_inverse() {
        let r = FieldElement::random_element();
        assert_eq!(
            ((r * r.inverse()).val + FieldElement::modulus()) % FieldElement::modulus(),
            1
        );
    }

    #[test]
    fn test_pow() {
        assert_eq!(
            FieldElement::generator().pow(FieldElement::modulus() - 1),
            FieldElement::one()
        );
    }

    #[test]
    fn test_is_order() {
        let g = FieldElement::generator().pow(3 * 2i128.pow(20));
        assert!(g.is_order(1024));
    }
}
