use super::field::FieldElement;
use super::list_utils;

#[derive(Clone, Debug)]
pub struct Polynomial {
    pub poly: Vec<FieldElement>,
    pub var: char,
}

impl Polynomial {
    pub fn new(coeffs: &[FieldElement], var: char) -> Self {
        let modified = list_utils::remove_trailing_elements(coeffs, &FieldElement::zero());
        Polynomial { poly: modified, var }
    }

    pub fn degree(&self) -> usize {
        if self.poly.is_empty() {
            return 0;
        }
        list_utils::remove_trailing_elements(&self.poly, &FieldElement::zero()).len().saturating_sub(1)
    }

    /// Horner's method evaluation.
    pub fn eval(&self, point: &FieldElement) -> FieldElement {
        self.poly
            .iter()
            .rev()
            .fold(FieldElement::zero(), |acc, coeff| acc * *point + *coeff)
    }

    pub fn compose(&self, other: &Polynomial) -> Polynomial {
        let mut res = Polynomial::new(&[], self.var);
        for coef in self.poly.iter().rev() {
            let coef_poly = Polynomial::new(&[*coef], self.var);
            res = (res * other.clone()) + coef_poly;
        }
        res
    }

    /// Polynomial long division returning (quotient, remainder).
    pub fn qdiv(&self, other: &Polynomial) -> (Polynomial, Polynomial) {
        let pol2 = &other.poly;
        assert!(!pol2.is_empty(), "Dividing by zero polynomial.");

        if self.poly.is_empty() {
            return (
                Polynomial::new(&[], self.var),
                Polynomial::new(&[], self.var),
            );
        }

        let mut rem = self.poly.clone();
        let mut deg_dif = rem.len() as i128 - pol2.len() as i128;
        let mut quotient = vec![FieldElement::zero(); (deg_dif + 1).max(0) as usize];

        let g_msc_inv = pol2.last().unwrap().inverse();

        while deg_dif >= 0 {
            let tmp = *rem.last().unwrap() * g_msc_inv;
            quotient[deg_dif as usize] = quotient[deg_dif as usize] + tmp;

            let mut last_non_zero = deg_dif - 1;
            for (i, coef) in pol2.iter().enumerate().map(|(idx, c)| (idx + deg_dif as usize, c))
            {
                rem[i] = rem[i] - (tmp * *coef);
                if rem[i] != FieldElement::zero() {
                    last_non_zero = i as i128;
                }
            }

            rem.truncate((last_non_zero + 1) as usize);
            deg_dif = rem.len() as i128 - pol2.len() as i128;
        }

        (
            Polynomial::new(&quotient, self.var),
            Polynomial::new(&rem, self.var),
        )
    }

    pub fn truediv(&self, other: &Polynomial) -> Polynomial {
        let (div, modulo) = self.qdiv(other);
        assert!(modulo == Polynomial::new(&[FieldElement::zero()], self.var));
        div
    }

    pub fn modulo(&self, other: &Polynomial) -> Polynomial {
        self.qdiv(other).1
    }

    pub fn scalar_mul(&self, scalar: &FieldElement) -> Polynomial {
        Polynomial {
            poly: list_utils::scalar_multiplication(&self.poly, scalar),
            var: self.var,
        }
    }

    pub fn monomial(degree: usize, coeff: &FieldElement) -> Polynomial {
        let mut coeffs = vec![FieldElement::zero(); degree];
        coeffs.push(*coeff);
        Polynomial { poly: coeffs, var: 'x' }
    }

    pub fn gen_linear_term(point: &FieldElement) -> Polynomial {
        Polynomial {
            poly: vec![-*point, FieldElement::one()],
            var: 'x',
        }
    }

    /// Binary exponentiation for polynomials.
    pub fn pow(&self, other: i128) -> Polynomial {
        let mut res = Polynomial::new(&[FieldElement::one()], 'x');
        let mut cur = self.clone();
        let mut exponent = other;
        loop {
            if exponent % 2 != 0 {
                res = res * cur.clone();
            }
            exponent >>= 1;
            if exponent == 0 {
                break;
            }
            cur = cur.clone() * cur;
        }
        res
    }

    pub fn calculate_lagrange_polynomials(x_values: &[FieldElement]) -> Vec<Polynomial> {
        let monomials: Vec<Polynomial> = x_values
            .iter()
            .map(|x| Polynomial::monomial(1, &FieldElement::one()) - Polynomial::monomial(0, x))
            .collect();

        let numerator = list_utils::prod(&monomials);

        let mut lagrange_polynomials = Vec::with_capacity(x_values.len());

        for j in 0..x_values.len() {
            let denominator = list_utils::prod(
                &x_values
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != j)
                    .map(|(_, x)| x_values[j] - *x)
                    .collect::<Vec<FieldElement>>(),
            );

            let (cur_poly, _) = numerator.qdiv(&monomials[j].scalar_mul(&denominator));
            lagrange_polynomials.push(cur_poly);
        }

        lagrange_polynomials
    }

    pub fn interpolate_poly_lagrange(
        y_values: &[FieldElement],
        lagrange_polynomials: &[Polynomial],
    ) -> Polynomial {
        let mut poly = Polynomial::new(&[], 'x');
        for (j, y_value) in y_values.iter().enumerate() {
            poly = poly + lagrange_polynomials[j].scalar_mul(y_value);
        }
        poly
    }

    pub fn interpolate_poly(
        x_values: &[FieldElement],
        y_values: &[FieldElement],
    ) -> Polynomial {
        assert_eq!(x_values.len(), y_values.len());
        let lagrange = Self::calculate_lagrange_polynomials(x_values);
        let poly = Self::interpolate_poly_lagrange(y_values, &lagrange);
        let ret = list_utils::remove_trailing_elements(&poly.poly, &FieldElement::zero());
        Polynomial { poly: ret, var: 'x' }
    }
}

impl PartialEq for Polynomial {
    fn eq(&self, other: &Self) -> bool {
        self.poly == other.poly
    }
}

impl std::ops::Add for Polynomial {
    type Output = Polynomial;
    fn add(self, rhs: Self) -> Self::Output {
        Polynomial {
            poly: list_utils::two_lists_tuple_operation(&self.poly, &rhs.poly, |a, b| *a + *b),
            var: self.var,
        }
    }
}

impl std::ops::Sub for Polynomial {
    type Output = Polynomial;
    fn sub(self, rhs: Self) -> Self::Output {
        Polynomial {
            poly: list_utils::two_lists_tuple_operation(&self.poly, &rhs.poly, |a, b| *a - *b),
            var: self.var,
        }
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Polynomial;
    fn neg(self) -> Self::Output {
        Polynomial::new(&[FieldElement::new(0)], self.var) - self
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Polynomial;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.poly.is_empty() || rhs.poly.is_empty() {
            return Polynomial::new(&[], self.var);
        }
        let res_len = self.poly.len() + rhs.poly.len() - 1;
        let mut res = vec![FieldElement::zero(); res_len];
        for (i, c1) in self.poly.iter().enumerate() {
            for (j, c2) in rhs.poly.iter().enumerate() {
                res[i + j] = res[i + j] + (*c1 * *c2);
            }
        }
        Polynomial { poly: res, var: self.var }
    }
}

impl std::ops::Div for Polynomial {
    type Output = Polynomial;
    fn div(self, rhs: Self) -> Self::Output {
        let (quotient, remainder) = self.qdiv(&rhs);
        assert!(remainder != Polynomial::from(0i128));
        quotient
    }
}

impl std::ops::Rem for Polynomial {
    type Output = Polynomial;
    fn rem(self, rhs: Self) -> Self::Output {
        self.qdiv(&rhs).1
    }
}

impl From<i128> for Polynomial {
    fn from(value: i128) -> Self {
        Polynomial {
            poly: vec![FieldElement::from(value)],
            var: 'x',
        }
    }
}

impl From<FieldElement> for Polynomial {
    fn from(value: FieldElement) -> Self {
        Polynomial { poly: vec![value], var: 'x' }
    }
}

impl Default for Polynomial {
    fn default() -> Self {
        Self {
            poly: vec![FieldElement::zero()],
            var: 'x',
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval() {
        // 5 + 3x + 2x^2
        let poly = Polynomial {
            poly: vec![
                FieldElement::from(5i128),
                FieldElement::from(3i128),
                FieldElement::from(2i128),
            ],
            var: 'x',
        };
        assert_eq!(poly.eval(&FieldElement::from(3i128)), FieldElement::from(32i128));
    }

    #[test]
    fn test_interpolate_roundtrip() {
        let x_values = vec![
            FieldElement::new(1),
            FieldElement::new(2),
            FieldElement::new(3),
            FieldElement::new(4),
        ];
        let original = Polynomial::new(
            &[
                FieldElement::new(5),
                FieldElement::new(3),
                FieldElement::new(2),
            ],
            'x',
        );
        let y_values: Vec<FieldElement> = x_values.iter().map(|x| original.eval(x)).collect();
        let interpolated = Polynomial::interpolate_poly(&x_values, &y_values);
        assert_eq!(interpolated, original);
    }

    #[test]
    fn test_qdiv() {
        let a = Polynomial::new(
            &[
                FieldElement::new(6),
                FieldElement::new(5),
                FieldElement::new(4),
                FieldElement::new(3),
            ],
            'x',
        );
        let b = Polynomial::new(&[FieldElement::new(1), FieldElement::new(2)], 'x');
        let (q, r) = a.qdiv(&b);
        // a == q*b + r
        let reconstructed = q * b + r;
        assert_eq!(reconstructed, a);
    }
}
