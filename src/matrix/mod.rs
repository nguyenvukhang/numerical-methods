mod algebra;
mod column_traits;
mod inits;
mod matrix_traits;
mod qr_decomposition;
mod scalar_traits;
mod square_matrix;

#[allow(unused)]
pub use algebra::*;
#[allow(unused)]
pub use column_traits::*;
#[allow(unused)]
pub use inits::*;
#[allow(unused)]
pub use matrix_traits::*;
#[allow(unused)]
pub use scalar_traits::*;

#[cfg(test)]
use crate::assert::*;

use rand::Rng;

use std::ops::RangeInclusive;

pub(crate) type R = f64;

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
        let mut rng = rand::thread_rng();
        Self::new().on_each(|_| rng.gen())
    }

    pub fn from(data: [[R; N]; M]) -> Self {
        Mat::<N, M> { data }.transpose()
    }

    pub fn transpose(&self) -> Mat<N, M> {
        let mut A = Mat::new();
        for i in self.row_iter() {
            for j in self.col_iter() {
                A[(j, i)] = self[(i, j)];
            }
        }
        A
    }

    /// (alias: tranpose())
    pub fn t(&self) -> Mat<N, M> {
        self.transpose()
    }

    pub fn eq<X: AsRef<Self>>(&self, rhs: X, abs_tol: R) -> bool {
        let rhs = rhs.as_ref();
        for i in self.row_iter() {
            for j in self.col_iter() {
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
        for j in self.col_iter() {
            m[(1, j)] = self[(i, j)];
        }
        m
    }

    /// Extract the `j`-th column of the matrix.
    pub fn col(&self, j: usize) -> Mat<M, 1> {
        Mat {
            data: [self.data[j - 1]],
        }
    }

    /// Set the `i`-th row of the matrix.
    pub fn set_row(&mut self, i: usize, row: Mat<1, N>) {
        self.col_iter().for_each(|j| self[(i, j)] = row[(1, j)]);
    }

    /// Set the `j`-th column of the matrix.
    pub fn set_col(&mut self, j: usize, col: Mat<M, 1>) {
        self.row_iter().for_each(|i| self[(i, j)] = col[i]);
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

    /// Apply a function element-wise
    pub fn on_each<F: FnMut(R) -> R>(&self, mut f: F) -> Self {
        let mut m = Self::new();
        for i in self.row_iter() {
            for j in self.col_iter() {
                m[(i, j)] = f(self[(i, j)]);
            }
        }
        m
    }

    pub fn on_each2<F: Fn(R, R) -> R>(&self, other: &Self, f: F) -> Self {
        let mut m = Self::new();
        for i in self.row_iter() {
            for j in self.col_iter() {
                m[(i, j)] = f(self[(i, j)], other[(i, j)]);
            }
        }
        m
    }

    /// Raise each element to a particular exponent.
    pub fn powf(&mut self, exponent: R) {
        self.on_each(|v| v.powf(exponent));
    }

    pub fn is_upper_triangular(&self) -> bool {
        for i in self.row_iter() {
            for j in 1..i {
                if self[(i, j)] != 0. {
                    return false;
                }
            }
        }
        true
    }

    /// Extracts the upper triangular portion of the matrix.
    pub fn upper_triangular(&self) -> Self {
        let mut m = self.clone();
        for i in self.row_iter() {
            for j in 1..i {
                m[(i, j)] = 0.;
            }
        }
        m
    }

    /// Extracts the lower triangular portion of the matrix.
    pub fn lower_triangular(&self) -> Self {
        let mut m = self.clone();
        for i in self.row_iter() {
            for j in i + 1..=N {
                m[(i, j)] = 0.;
            }
        }
        m
    }

    /// Extracts the top n×n sub-matrix.
    pub fn top_square(&self) -> Mat<N, N> {
        self.top_n_rows::<N>()
    }

    pub fn top_n_rows<const U: usize>(&self) -> Mat<U, N> {
        let mut m = Mat::<U, N>::new();
        assert!(
            m.nrows() <= self.nrows(),
            "Not enough rows in matrix to take first {}",
            m.nrows()
        );
        for i in m.row_iter() {
            for j in m.col_iter() {
                m[(i, j)] = self[(i, j)];
            }
        }
        m
    }

    /// Swaps columns `a` and `b` in the matrix.
    pub fn swap_columns(&mut self, a: usize, b: usize) {
        self.data.swap(a, b);
    }

    /// Returns true if none of the entries of this matrix is NaN.
    pub fn contains_nan(&self) -> bool {
        for i in self.row_iter() {
            for j in self.col_iter() {
                if self[(i, j)].is_nan() {
                    return true;
                }
            }
        }
        false
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
}

#[test]
fn solve_qr_test() {
    for _ in 0..100 {
        let A = Mat::<6, 4>::rand();
        let b = Mat::<6, 1>::rand();
        let x = A.solve_lls(&b);
        let AT = A.transpose();
        assert_eq_mat(&AT * A * x, &AT * b, 1e-10);
    }
}
