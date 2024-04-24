mod algebra;
mod column_methods;
mod core_traits;
mod inits;
mod scalar_traits;
mod square_matrix;

pub use inits::*;
pub use scalar_traits::*;

use crate::na;
use crate::prelude::*;

use std::ops::RangeInclusive;

/// 1-indexed column-major matrix data structure.
///
/// 1-indexed to stay in line with notation in commmon Math texts.
/// column-major to easily take linear combination of columns.
#[derive(Clone)]
pub struct Mat<const M: usize, const N: usize> {
    pub data: [[R; M]; N],
}

impl<const M: usize, const N: usize> Mat<M, N> {
    /// New matrix (of zeros).
    pub fn new() -> Self {
        Self::zero()
    }

    /// Matrix of zeros.
    pub fn zero() -> Self {
        Self { data: [[0.; M]; N] }
    }

    /// Generate a matrix populated with random values between 0 and 1.
    pub fn rand() -> Self {
        Self::new().on_each(|_| rand::random())
    }

    pub fn from(data: [[R; N]; M]) -> Self {
        Mat { data }.transpose()
    }

    pub fn transpose(&self) -> Mat<N, M> {
        let mut A = Mat::new();
        (1..=M).for_each(|i| (1..=N).for_each(|j| A[(j, i)] = self[(i, j)]));
        A
    }

    /// (alias: tranpose())
    pub fn t(&self) -> Mat<N, M> {
        self.transpose()
    }

    pub fn eq<X: AsRef<Self>>(&self, rhs: X, abs_tol: R) -> bool {
        let rhs = rhs.as_ref();
        for i in 1..=M {
            for j in 1..=N {
                if self[(i, j)].abs_diff(rhs[(i, j)]) > abs_tol {
                    return false;
                }
            }
        }
        true
    }

    pub fn row_iter(&self) -> RangeInclusive<usize> {
        1..=M
    }

    pub fn col_iter(&self) -> RangeInclusive<usize> {
        1..=N
    }

    /// Extract the `i`-th row of the matrix.
    pub fn row(&self, i: usize) -> Mat<1, N> {
        let mut m = Mat::new();
        (1..=N).for_each(|j| m[(1, j)] = self[(i, j)]);
        m
    }

    pub fn col_raw(&self, j: usize) -> &[R; M] {
        &self.data[j - 1]
    }

    /// Extract the `j`-th column of the matrix.
    pub fn col(&self, j: usize) -> &Mat<M, 1> {
        let colref = self.data.get(j - 1).unwrap();
        unsafe { std::mem::transmute(colref) }
    }

    /// Get mutable reference to the `j`-th column of the matrix.
    pub fn col_mut(&mut self, j: usize) -> &mut Mat<M, 1> {
        let colref = self.data.get_mut(j - 1).unwrap();
        unsafe { std::mem::transmute(colref) }
    }

    /// Set the `i`-th row of the matrix.
    pub fn set_row(&mut self, i: usize, row: Mat<1, N>) {
        (1..=N).for_each(|j| self[(i, j)] = row[(1, j)]);
    }

    /// Set the `j`-th column of the matrix.
    pub fn set_col(&mut self, j: usize, col: Mat<M, 1>) {
        self.data[j - 1] = col.data[0];
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (M, N)
    }

    pub fn nrows(&self) -> usize {
        M
    }

    pub fn ncols(&self) -> usize {
        N
    }

    /// Apply a function element-wise, and create a new matrix.
    pub fn on_each<F: FnMut(R) -> R>(&self, mut f: F) -> Self {
        let mut m = Self::new();
        for i in 1..=M {
            for j in 1..=N {
                m[(i, j)] = f(self[(i, j)]);
            }
        }
        m
    }

    /// Apply a function element-wise.
    pub fn on_each_mut<F: FnMut(R) -> R>(&mut self, mut f: F) {
        for i in 1..=M {
            for j in 1..=N {
                self[(i, j)] = f(self[(i, j)]);
            }
        }
    }

    /// Zip up two same-dimesion matrices with a function.
    pub fn on_each2<F: Fn(R, R) -> R>(&self, other: &Self, f: F) -> Self {
        let mut m = Self::new();
        for i in 1..=M {
            for j in 1..=N {
                m[(i, j)] = f(self[(i, j)], other[(i, j)]);
            }
        }
        m
    }

    /// Raise each element to a particular exponent.
    pub fn powf(&mut self, x: R) {
        (1..=M).for_each(|i| {
            (1..=N).for_each(|j| self[(i, j)] = self[(i, j)].powf(x))
        });
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in 1..=M {
            for j in 1..i {
                if self[(i, j)] != 0. {
                    return false;
                }
            }
        }
        true
    }

    /// Zeros-out entries to become upper triangular.
    pub fn to_upper_triangular(&mut self) {
        (1..=M).for_each(|i| (1..i).for_each(|j| self[(i, j)] = 0.));
    }

    /// Zeros-out entries to become upper triangular.
    pub fn to_lower_triangular(&mut self) {
        (1..=M).for_each(|i| (i + 1..=N).for_each(|j| self[(i, j)] = 0.));
    }

    /// Extracts the upper triangular portion of the matrix.
    pub fn upper_triangular(&self) -> Self {
        let mut m = self.clone();
        m.to_upper_triangular();
        m
    }

    /// Extracts the lower triangular portion of the matrix.
    pub fn lower_triangular(&self) -> Self {
        let mut m = self.clone();
        m.to_lower_triangular();
        m
    }

    /// Extracts the top n×n sub-matrix.
    pub fn top_square(&self) -> Mat<N, N> {
        self.top_n_rows()
    }

    pub fn top_n_rows<const U: usize>(&self) -> Mat<U, N> {
        let mut m = Mat::new();
        assert!(
            m.nrows() <= self.nrows(),
            "Not enough rows in matrix to take first {}",
            m.nrows()
        );
        (1..=U).for_each(|i| (1..=N).for_each(|j| m[(i, j)] = self[(i, j)]));
        m
    }

    /// Swaps columns `a` and `b` in the matrix.
    pub fn swap_columns(&mut self, a: usize, b: usize) {
        self.data.swap(a, b);
    }

    /// Returns true if none of the entries of this matrix is NaN.
    pub fn contains_nan(&self) -> bool {
        for i in 1..=M {
            for j in 1..=N {
                if self[(i, j)].is_nan() {
                    return true;
                }
            }
        }
        false
    }

    /// Execute a QR decomposition via Householder reflections.
    /// This requires M ≥ N.
    pub fn qr_householder(&self) -> (Mat<M, M>, Mat<M, N>) {
        na::qr_decomp::householder(self)
    }

    /// Solve linear-least-squares.
    /// Decomposes `self` into QR via householder, then applied backsub.
    pub fn solve_lls(&self, b: &Mat<M, 1>) -> Mat<N, 1> {
        let (Q, R) = self.qr_householder();
        let v = Q.transpose() * b;
        let R1 = R.top_n_rows::<N>();
        let v1 = v.top_n_rows::<N>();
        R1.backward_sub(&v1)
    }

    /// For column vectors, this gives the l1-norm or Manhattan
    /// distance or the Taxicab norm.
    /// For matrices, this gives the operator's l1-norm.
    pub fn l1_norm(&self) -> R {
        let mut max = 0.;
        for j in 1..=N {
            let s = self.col_raw(j).iter().map(|v| v.abs()).sum();
            if s > max {
                max = s;
            }
        }
        max
    }

    /// For column vectors, this gives the l2-norm or Euclidean norm.
    /// For matrices, this gives the Spectral norm, the square root of
    /// the largest eigenvalue of AᵀA.
    pub fn l2_norm(&self) -> R {
        match N {
            1 => {
                let v = self.col(1); // the one and only column.
                v.dot(v).sqrt()
            }
            _ => {
                let (lambda, _) = na::power_iteration(&(self.t() * self));
                lambda.sqrt()
            }
        }
    }
}

#[test]
fn solve_qr_test() {
    for _ in 0..REPS {
        let A = Mat::<6, 4>::rand();
        let b = Mat::<6, 1>::rand();
        let x = A.solve_lls(&b);
        let AT = A.transpose();
        assert_eq_mat!(&AT * A * x, AT * b, 1e-10);
    }
}
