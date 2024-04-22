use super::{Mat, R};

use std::fmt;
use std::ops::{Index, IndexMut, Neg};

impl<const M: usize, const N: usize> fmt::Display for Mat<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix")?;
        for i in 1..=M {
            write!(f, "  ")?;
            for j in self.col_iter() {
                write!(f, "{:>8.4}", self[(i, j)])?;
            }
            if i < M {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
impl<const M: usize, const N: usize> fmt::Debug for Mat<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl<const M: usize, const N: usize> Index<(usize, usize)> for Mat<M, N> {
    type Output = R;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.1 - 1][index.0 - 1]
    }
}

impl<const M: usize, const N: usize> IndexMut<(usize, usize)> for Mat<M, N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.1 - 1][index.0 - 1]
    }
}

/// Indexing column vectors
/// (RIP row vectors because rust complains on the overlap of N).
impl<const N: usize> Index<usize> for Mat<N, 1> {
    type Output = R;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[0][index - 1]
    }
}

/// Indexing column vectors
/// (RIP row vectors because rust complains on the overlap of N).
impl<const N: usize> IndexMut<usize> for Mat<N, 1> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[0][index - 1]
    }
}

impl<const M: usize, const N: usize> AsRef<Mat<M, N>> for Mat<M, N> {
    fn as_ref(&self) -> &Mat<M, N> {
        &self
    }
}

impl<const M: usize, const N: usize> Neg for &Mat<M, N> {
    type Output = Mat<M, N>;
    fn neg(self) -> Self::Output {
        self.on_each(|v| -v)
    }
}

impl<const M: usize, const N: usize> Neg for Mat<M, N> {
    type Output = Mat<M, N>;
    fn neg(self) -> Self::Output {
        self.on_each(|v| -v)
    }
}
