use super::{Mat, R};

use std::ops::{Div, DivAssign, Mul, MulAssign};

/// Convert a 1x1 matrix into a scalar.
impl From<Mat<1, 1>> for R {
    fn from(value: Mat<1, 1>) -> Self {
        value[(1, 1)]
    }
}

pub trait RealTraits {
    fn abs_diff(&self, rhs: Self) -> R;
    fn rel_diff(&self, rhs: Self) -> R;
}

impl RealTraits for R {
    fn abs_diff(&self, rhs: Self) -> R {
        if self > &rhs {
            self - rhs
        } else {
            rhs - self
        }
    }

    fn rel_diff(&self, rhs: Self) -> R {
        let d = self.abs_diff(rhs);
        let b = self.abs().max(rhs.abs());
        d / b
    }
}

/// Core matrix scalar multiplication. All other implementations will call this.
impl<const M: usize, const N: usize> MulAssign<R> for Mat<M, N> {
    fn mul_assign(&mut self, x: R) {
        (1..=M).for_each(|i| (1..=N).for_each(|j| self[(i, j)] *= x));
    }
}

impl<const M: usize, const N: usize> Mul<R> for Mat<M, N> {
    type Output = Mat<M, N>;
    fn mul(mut self, x: R) -> Self::Output {
        self *= x;
        self
    }
}

impl<const M: usize, const N: usize> Mul<Mat<M, N>> for R {
    type Output = Mat<M, N>;
    fn mul(self, m: Mat<M, N>) -> Self::Output {
        m * self
    }
}

impl<const M: usize, const N: usize> Mul<R> for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn mul(self, x: R) -> Self::Output {
        self.clone() * x
    }
}

impl<const M: usize, const N: usize> Mul<&Mat<M, N>> for R {
    type Output = Mat<M, N>;
    fn mul(self, m: &Mat<M, N>) -> Self::Output {
        m * self
    }
}

/// Core matrix scalar division. All other implementations will call this.
impl<const M: usize, const N: usize> DivAssign<R> for Mat<M, N> {
    fn div_assign(&mut self, x: R) {
        (1..=M).for_each(|i| (1..=N).for_each(|j| self[(i, j)] /= x));
    }
}

impl<const M: usize, const N: usize> Div<R> for Mat<M, N> {
    type Output = Mat<M, N>;
    fn div(mut self, x: R) -> Self::Output {
        self /= x;
        self
    }
}

impl<const M: usize, const N: usize> Div<R> for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn div(self, x: R) -> Self::Output {
        self.clone() / x
    }
}
