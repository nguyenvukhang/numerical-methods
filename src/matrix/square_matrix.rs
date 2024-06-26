use super::*;

impl<const N: usize> Mat<N, N> {
    /// Apply backward substitution on Ax = b, with A := self.
    pub fn backward_sub(&self, b: &Mat<N, 1>) -> Mat<N, 1> {
        assert!(
            self.is_upper_triangular(),
            "A needs to be upper-triangular:\n{self}"
        );
        let mut x = b.clone();
        for k in (1..=N).rev() {
            for j in k + 1..=N {
                x[k] -= self[(k, j)] * x[j];
            }
            x[k] /= self[(k, k)];
        }
        x
    }

    /// Tranpose in-place; possible since it's a square.
    pub fn transpose_inplace(&mut self) {
        use std::ptr;
        for i in 1..=N {
            for j in 1..i {
                let a = ptr::addr_of_mut!(self[(i, j)]);
                let b = ptr::addr_of_mut!(self[(j, i)]);
                unsafe { ptr::swap(a, b) }
            }
        }
    }

    /// Trace: sum of elements on the diagonal
    pub fn trace(&self) -> R {
        (1..=N).map(|i| self[(i, i)]).sum()
    }

    /// Create a new identity matrix.
    pub fn eye() -> Self {
        Self::from_fn(|i, j| R::from(i == j))
    }

    /// Adds `lambda` * `I` to `self`.
    pub fn add_identity(&mut self, lambda: R) {
        (1..=N).for_each(|i| self[(i, i)] += lambda);
    }

    /// Create a random symmetric matrix.
    pub fn symmetric() -> Self {
        let x = Mat::rand();
        (&x + x.transpose()) / 2.
    }

    /// Create a random symmetric positive definite matrix.
    pub fn symmetric_positive_definite() -> Self {
        Mat::symmetric() + Mat::eye()
    }
}
