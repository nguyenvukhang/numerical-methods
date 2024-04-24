use super::Mat;
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

/// Core matrix negation. All other implementations will call this.
impl<const M: usize, const N: usize> Neg for Mat<M, N> {
    type Output = Mat<M, N>;
    fn neg(mut self) -> Self::Output {
        for i in 1..=M {
            (1..=N).for_each(|j| self[(i, j)] = -self[(i, j)]);
        }
        self
    }
}

impl<const M: usize, const N: usize> Neg for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

/// Core matrix addition. All other implementations will call this.
impl<const M: usize, const N: usize> AddAssign<&Mat<M, N>> for &mut Mat<M, N> {
    fn add_assign(&mut self, B: &Mat<M, N>) {
        (1..=M).for_each(|i| (1..=N).for_each(|j| self[(i, j)] += B[(i, j)]));
    }
}

impl<const M: usize, const N: usize> AddAssign<Mat<M, N>> for &mut Mat<M, N> {
    fn add_assign(&mut self, B: Mat<M, N>) {
        *self += &B;
    }
}

impl<const M: usize, const N: usize> Add<&Mat<M, N>> for Mat<M, N> {
    type Output = Mat<M, N>;
    fn add(mut self, B: &Mat<M, N>) -> Self::Output {
        let mut x = &mut self;
        x += B;
        self
    }
}

impl<const M: usize, const N: usize> Add<Mat<M, N>> for Mat<M, N> {
    type Output = Mat<M, N>;
    fn add(self, rhs: Mat<M, N>) -> Self::Output {
        self + &rhs
    }
}

impl<const M: usize, const N: usize> Add<Mat<M, N>> for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn add(self, rhs: Mat<M, N>) -> Self::Output {
        rhs + self
    }
}

impl<const M: usize, const N: usize> Add<&Mat<M, N>> for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn add(self, rhs: &Mat<M, N>) -> Self::Output {
        self.clone() + rhs
    }
}

/// Core matrix subtraction. All other implementations will call this.
impl<const M: usize, const N: usize> SubAssign<&Mat<M, N>> for &mut Mat<M, N> {
    fn sub_assign(&mut self, B: &Mat<M, N>) {
        (1..=M).for_each(|i| (1..=N).for_each(|j| self[(i, j)] -= B[(i, j)]));
    }
}

impl<const M: usize, const N: usize> SubAssign<Mat<M, N>> for &mut Mat<M, N> {
    fn sub_assign(&mut self, B: Mat<M, N>) {
        *self -= &B;
    }
}

impl<const M: usize, const N: usize> Sub<&Mat<M, N>> for Mat<M, N> {
    type Output = Mat<M, N>;
    fn sub(mut self, B: &Mat<M, N>) -> Self::Output {
        let mut x = &mut self;
        x -= B;
        self
    }
}

impl<const M: usize, const N: usize> Sub<Mat<M, N>> for Mat<M, N> {
    type Output = Mat<M, N>;
    fn sub(self, rhs: Mat<M, N>) -> Self::Output {
        self - &rhs
    }
}

impl<const M: usize, const N: usize> Sub<Mat<M, N>> for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn sub(self, rhs: Mat<M, N>) -> Self::Output {
        -rhs + self
    }
}

impl<const M: usize, const N: usize> Sub<&Mat<M, N>> for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn sub(self, rhs: &Mat<M, N>) -> Self::Output {
        self.clone() - rhs
    }
}

/// Core matrix multiplication. All other implementations will call this.
impl<const M: usize, const N: usize, const P: usize> Mul<&Mat<P, N>>
    for &Mat<M, P>
{
    type Output = Mat<M, N>;
    fn mul(self, rhs: &Mat<P, N>) -> Self::Output {
        let mut m = Mat::zero(); // new allocation is inevitable due to dimensions.
        for i in 1..=M {
            for j in 1..=N {
                (1..=P).for_each(|k| m[(i, j)] += self[(i, k)] * rhs[(k, j)]);
            }
        }
        m
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Mat<P, N>>
    for Mat<M, P>
{
    type Output = Mat<M, N>;
    fn mul(self, rhs: Mat<P, N>) -> Self::Output {
        &self * &rhs
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<Mat<P, N>>
    for &Mat<M, P>
{
    type Output = Mat<M, N>;
    fn mul(self, rhs: Mat<P, N>) -> Self::Output {
        self * &rhs
    }
}

impl<const M: usize, const N: usize, const P: usize> Mul<&Mat<P, N>>
    for Mat<M, P>
{
    type Output = Mat<M, N>;
    fn mul(self, rhs: &Mat<P, N>) -> Self::Output {
        &self * rhs
    }
}
