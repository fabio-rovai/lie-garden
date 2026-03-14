use nalgebra::DMatrix;

/// Supported Lie group types.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GroupType {
    /// Special Orthogonal — rotations, det=1, orthogonal
    SO,
    /// Special Euclidean — rigid motions (rotation + translation)
    SE,
    /// General Linear — all invertible matrices
    GL,
}

/// A Lie group instance with a specific type and dimension.
#[derive(Debug, Clone)]
pub struct LieGroup {
    pub group_type: GroupType,
    pub n: usize,
}

/// A group element — an n*n matrix.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GroupElement {
    data: Vec<f64>,
    n: usize,
}

impl GroupElement {
    pub fn from_matrix(m: &DMatrix<f64>) -> Self {
        Self {
            data: m.as_slice().to_vec(),
            n: m.nrows(),
        }
    }

    pub fn to_matrix(&self) -> DMatrix<f64> {
        DMatrix::from_column_slice(self.n, self.n, &self.data)
    }

    pub fn matrix(&self) -> &[f64] {
        &self.data
    }

    pub fn dim(&self) -> usize {
        self.n
    }

    /// Check if two group elements are close within tolerance.
    pub fn close_to(&self, other: &GroupElement, tol: f64) -> bool {
        self.data.len() == other.data.len()
            && self.data.iter().zip(other.data.iter()).all(|(a, b)| (a - b).abs() < tol)
    }
}

impl LieGroup {
    pub fn new(group_type: GroupType, n: usize) -> Self {
        Self { group_type, n }
    }

    /// Dimension of the Lie algebra.
    pub fn algebra_dim(&self) -> usize {
        match self.group_type {
            GroupType::SO => self.n * (self.n - 1) / 2,
            GroupType::SE => {
                // SE uses n×n homogeneous matrices. n is the matrix dim, not the space dim.
                // SE(k) in standard math notation uses (k+1)×(k+1) matrices, so pass n=k+1.
                // Example: SE(3) → n=4, rot in SO(3) dim=3, translation dim=3, total=6.
                // Example: SE(2) → n=3, rot in SO(2) dim=1, translation dim=2, total=3.
                let rot_n = self.n - 1;
                rot_n * (rot_n - 1) / 2 + (self.n - 1)
            }
            GroupType::GL => self.n * self.n,
        }
    }

    /// Matrix dimension.
    pub fn matrix_dim(&self) -> usize {
        self.n
    }

    /// Identity element.
    pub fn identity(&self) -> GroupElement {
        let dim = self.matrix_dim();
        GroupElement::from_matrix(&DMatrix::identity(dim, dim))
    }

    /// Group multiplication.
    pub fn multiply(&self, a: &GroupElement, b: &GroupElement) -> GroupElement {
        let ma = a.to_matrix();
        let mb = b.to_matrix();
        let result = &ma * &mb;
        self.project_to_group(&GroupElement::from_matrix(&result))
    }

    /// Group inverse.
    pub fn inverse(&self, a: &GroupElement) -> GroupElement {
        let m = a.to_matrix();
        match self.group_type {
            GroupType::SO => {
                GroupElement::from_matrix(&m.transpose())
            }
            GroupType::GL => {
                let inv = m.clone().try_inverse().unwrap_or_else(|| DMatrix::identity(self.n, self.n));
                GroupElement::from_matrix(&inv)
            }
            GroupType::SE => {
                // SE(n) inverse: [R|t]^-1 = [R^T | -R^T * t]
                let n = self.n;
                let rot_n = n - 1;
                let mut inv_m = DMatrix::identity(n, n);
                // Extract and transpose rotation block
                let rot = m.view((0, 0), (rot_n, rot_n));
                let rot_t = rot.transpose();
                for i in 0..rot_n {
                    for j in 0..rot_n {
                        inv_m[(i, j)] = rot_t[(i, j)];
                    }
                }
                // Translation: -R^T * t
                let t_col = m.column(n - 1);
                let t = t_col.rows(0, rot_n);
                let neg_rt_t = -&rot_t * t;
                for i in 0..rot_n {
                    inv_m[(i, n - 1)] = neg_rt_t[i];
                }
                GroupElement::from_matrix(&inv_m)
            }
        }
    }

    /// Exponential map: Lie algebra -> Lie group.
    /// Takes algebra coefficients, returns on-group element.
    /// THIS IS THE CORE CONFINEMENT GUARANTEE.
    pub fn exp(&self, coefficients: &[f64]) -> GroupElement {
        let algebra_matrix = self.coefficients_to_algebra_matrix(coefficients);
        let result = match self.group_type {
            GroupType::SO if self.n == 3 => so3_exp(&algebra_matrix),
            _ => matrix_exp(&algebra_matrix),
        };
        self.project_to_group(&GroupElement::from_matrix(&result))
    }

    /// Logarithmic map: Lie group -> Lie algebra.
    pub fn log(&self, element: &GroupElement) -> Vec<f64> {
        let m = element.to_matrix();
        let log_m = match self.group_type {
            GroupType::SO if self.n == 3 => so3_log(&m),
            _ => matrix_log(&m),
        };
        self.algebra_matrix_to_coefficients(&log_m)
    }

    /// Check if element is a valid group member.
    pub fn is_member(&self, element: &GroupElement) -> bool {
        let m = element.to_matrix();
        match self.group_type {
            GroupType::SO => {
                let mt_m = m.transpose() * &m;
                let id = DMatrix::identity(self.n, self.n);
                let ortho_err = (&mt_m - &id).norm();
                let det_err = (m.determinant() - 1.0).abs();
                ortho_err < 1e-6 && det_err < 1e-6
            }
            GroupType::GL => {
                m.determinant().abs() > 1e-10
            }
            GroupType::SE => {
                let n = self.n;
                let rot_n = n - 1;
                // Check rotation block is in SO(rot_n)
                let rot = m.view((0, 0), (rot_n, rot_n));
                let rot_m = rot.clone_owned();
                let rt_r = rot_m.transpose() * &rot_m;
                let id = DMatrix::identity(rot_n, rot_n);
                let ortho_err = (&rt_r - &id).norm();
                let det_err = (rot_m.determinant() - 1.0).abs();
                if ortho_err > 1e-6 || det_err > 1e-6 {
                    return false;
                }
                // Check last row is [0...0, 1]
                for j in 0..rot_n {
                    if m[(n - 1, j)].abs() > 1e-6 {
                        return false;
                    }
                }
                (m[(n - 1, n - 1)] - 1.0).abs() < 1e-6
            }
        }
    }

    /// Project a matrix back onto the group (SVD-based).
    pub fn project_to_group(&self, element: &GroupElement) -> GroupElement {
        let m = element.to_matrix();
        match self.group_type {
            GroupType::SO => {
                let svd = m.svd(true, true);
                let u = svd.u.unwrap();
                let vt = svd.v_t.unwrap();
                let mut r = &u * &vt;
                if r.determinant() < 0.0 {
                    let mut u_fixed = u;
                    let last_col = u_fixed.ncols() - 1;
                    for i in 0..u_fixed.nrows() {
                        u_fixed[(i, last_col)] *= -1.0;
                    }
                    r = &u_fixed * &vt;
                }
                GroupElement::from_matrix(&r)
            }
            GroupType::GL => {
                if m.determinant().abs() < 1e-10 {
                    return self.identity();
                }
                element.clone()
            }
            GroupType::SE => {
                // Project rotation block to SO(n-1), preserve translation, fix last row
                let n = self.n;
                let rot_n = n - 1;
                let mut result = DMatrix::identity(n, n);

                // SVD-project rotation block onto SO(rot_n)
                let rot = m.view((0, 0), (rot_n, rot_n)).clone_owned();
                let svd = rot.svd(true, true);
                let u = svd.u.unwrap();
                let vt = svd.v_t.unwrap();
                let mut r = &u * &vt;
                if r.determinant() < 0.0 {
                    let mut u_fixed = u;
                    let last_col = u_fixed.ncols() - 1;
                    for i in 0..u_fixed.nrows() {
                        u_fixed[(i, last_col)] *= -1.0;
                    }
                    r = &u_fixed * &vt;
                }
                for i in 0..rot_n {
                    for j in 0..rot_n {
                        result[(i, j)] = r[(i, j)];
                    }
                }

                // Preserve translation column
                for i in 0..rot_n {
                    result[(i, n - 1)] = m[(i, n - 1)];
                }
                // Last row is already [0...0, 1] from identity init

                GroupElement::from_matrix(&result)
            }
        }
    }

    /// Convert algebra coefficients to the algebra matrix representation.
    pub fn coefficients_to_algebra_matrix(&self, coefficients: &[f64]) -> DMatrix<f64> {
        match self.group_type {
            GroupType::SO => {
                let mut m = DMatrix::zeros(self.n, self.n);
                let mut idx = 0;
                for i in 0..self.n {
                    for j in (i + 1)..self.n {
                        if idx < coefficients.len() {
                            m[(i, j)] = coefficients[idx];
                            m[(j, i)] = -coefficients[idx];
                            idx += 1;
                        }
                    }
                }
                m
            }
            GroupType::GL => {
                let mut m = DMatrix::zeros(self.n, self.n);
                for (idx, &c) in coefficients.iter().enumerate() {
                    if idx < self.n * self.n {
                        m[(idx / self.n, idx % self.n)] = c;
                    }
                }
                m
            }
            GroupType::SE => {
                let rot_n = self.n - 1;
                let rot_dim = rot_n * (rot_n - 1) / 2;
                let mut m = DMatrix::zeros(self.n, self.n);
                // Skew-symmetric rotation block
                let mut idx = 0;
                for i in 0..rot_n {
                    for j in (i + 1)..rot_n {
                        if idx < coefficients.len() {
                            m[(i, j)] = coefficients[idx];
                            m[(j, i)] = -coefficients[idx];
                            idx += 1;
                        }
                    }
                }
                // Translation in last column
                for i in 0..rot_n {
                    if rot_dim + i < coefficients.len() {
                        m[(i, self.n - 1)] = coefficients[rot_dim + i];
                    }
                }
                // Last row all zeros (se(n) structure)
                m
            }
        }
    }

    /// Convert algebra matrix back to coefficients.
    pub fn algebra_matrix_to_coefficients(&self, m: &DMatrix<f64>) -> Vec<f64> {
        match self.group_type {
            GroupType::SO => {
                let mut coeffs = Vec::new();
                for i in 0..self.n {
                    for j in (i + 1)..self.n {
                        coeffs.push(m[(i, j)]);
                    }
                }
                coeffs
            }
            GroupType::GL => {
                let mut coeffs = Vec::with_capacity(self.n * self.n);
                for i in 0..self.n {
                    for j in 0..self.n {
                        coeffs.push(m[(i, j)]);
                    }
                }
                coeffs
            }
            GroupType::SE => {
                let rot_n = self.n - 1;
                let mut coeffs = Vec::new();
                for i in 0..rot_n {
                    for j in (i + 1)..rot_n {
                        coeffs.push(m[(i, j)]);
                    }
                }
                for i in 0..rot_n {
                    coeffs.push(m[(i, self.n - 1)]);
                }
                coeffs
            }
        }
    }
}

/// Exact SO(3) exponential via Rodrigues formula.
/// exp([w]_x) = I + sin(theta)/theta * [w]_x + (1 - cos(theta))/theta^2 * [w]_x^2
fn so3_exp(skew: &DMatrix<f64>) -> DMatrix<f64> {
    let id = DMatrix::<f64>::identity(3, 3);
    // theta = ||w|| where skew = [w]_x
    let w1 = skew[(2, 1)]; // w_x
    let w2 = skew[(0, 2)]; // w_y
    let w3 = skew[(1, 0)]; // w_z
    let theta = (w1 * w1 + w2 * w2 + w3 * w3).sqrt();

    if theta < 1e-14 {
        return id;
    }

    let s = theta.sin() / theta;
    let c = (1.0 - theta.cos()) / (theta * theta);
    let skew2 = skew * skew;

    &id + skew * s + &skew2 * c
}

/// Exact SO(3) logarithm via Rodrigues formula.
fn so3_log(r: &DMatrix<f64>) -> DMatrix<f64> {
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta.abs() < 1e-10 {
        return (r - r.transpose()) * 0.5;
    }

    if (std::f64::consts::PI - theta).abs() < 1e-6 {
        let s = (r + DMatrix::identity(3, 3)) * 0.5;
        let mut best_col = 0;
        let mut best_val = s[(0, 0)];
        for i in 1..3 {
            if s[(i, i)] > best_val {
                best_val = s[(i, i)];
                best_col = i;
            }
        }
        let mut v = s.column(best_col).into_owned();
        let norm = v.norm();
        if norm > 1e-10 {
            v /= norm;
        }
        let mut result = DMatrix::zeros(3, 3);
        result[(0, 1)] = -v[2] * theta;
        result[(1, 0)] = v[2] * theta;
        result[(0, 2)] = v[1] * theta;
        result[(2, 0)] = -v[1] * theta;
        result[(1, 2)] = -v[0] * theta;
        result[(2, 1)] = v[0] * theta;
        return result;
    }

    let factor = theta / (2.0 * theta.sin());
    (r - r.transpose()) * factor
}

/// Matrix exponential via (6,6) Padé approximant with scaling and squaring.
///
/// Coefficients: b_j = (12-j)! 6! / (12! j! (6-j)!) for p=6.
/// Error at ||X|| ≤ 0.5: O(x^13) ≈ 2e-14 (near machine precision).
fn matrix_exp(m: &DMatrix<f64>) -> DMatrix<f64> {
    let n = m.nrows();
    let norm = m.norm();

    let s = (norm / 0.5).log2().ceil().max(0.0) as u32;
    let a = m / (2.0_f64.powi(s as i32));

    let id = DMatrix::<f64>::identity(n, n);
    let a2 = &a * &a;
    let a4 = &a2 * &a2;
    let a6 = &a4 * &a2;

    // (6,6) Padé coefficients
    const B: [f64; 7] = [
        1.0,
        0.5,
        5.0 / 44.0,       // b_2
        1.0 / 66.0,       // b_3
        1.0 / 792.0,      // b_4
        1.0 / 15840.0,    // b_5
        1.0 / 665280.0,   // b_6
    ];

    // Even part: V = b_0 I + b_2 A² + b_4 A⁴ + b_6 A⁶
    let v = &id * B[0] + &a2 * B[2] + &a4 * B[4] + &a6 * B[6];
    // Odd part: U = A(b_1 I + b_3 A² + b_5 A⁴)
    let u = &a * (&id * B[1] + &a2 * B[3] + &a4 * B[5]);

    // exp(A) ≈ (V - U)⁻¹ (V + U)
    let numer = &v + &u;
    let denom = &v - &u;

    let mut result = denom.try_inverse().unwrap_or(id.clone()) * numer;

    for _ in 0..s {
        result = &result * &result;
    }

    result
}

/// Matrix logarithm via inverse scaling and squaring.
///
/// Uses 11-term Taylor series: log(I+X) = X - X²/2 + X³/3 - ... + X¹¹/11
/// At ||X|| ≤ 0.5: truncation error ≈ 0.5^12/12 ≈ 2e-5 per sqrt level.
/// Combined with scaling (||X|| ≤ 0.1 after enough sqrts), error ≈ 1e-12.
fn matrix_log(m: &DMatrix<f64>) -> DMatrix<f64> {
    let id = DMatrix::<f64>::identity(m.nrows(), m.nrows());

    let mut a = m.clone();
    let mut k = 0u32;
    while (&a - &id).norm() > 0.1 && k < 30 {
        a = matrix_sqrt(&a);
        k += 1;
    }

    // Taylor series: log(I+X) = Σ (-1)^(n+1) X^n / n for n=1..11
    let x = &a - &id;
    let mut power = x.clone();
    let mut result = x.clone();
    for n in 2..=11u32 {
        power = &power * &x;
        let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
        result += &power * (sign / n as f64);
    }

    result * 2.0_f64.powi(k as i32)
}

/// Matrix square root via Denman-Beavers iteration.
fn matrix_sqrt(m: &DMatrix<f64>) -> DMatrix<f64> {
    let n = m.nrows();
    let id = DMatrix::<f64>::identity(n, n);
    let mut y = m.clone();
    let mut z = id.clone();

    for _ in 0..20 {
        let y_inv = y.clone().try_inverse().unwrap_or(id.clone());
        let z_inv = z.clone().try_inverse().unwrap_or(id.clone());
        let y_new = (&y + &z_inv) * 0.5;
        let z_new = (&z + &y_inv) * 0.5;
        let err = (&y_new - &y).norm();
        y = y_new;
        z = z_new;
        if err < 1e-12 {
            break;
        }
    }
    y
}
