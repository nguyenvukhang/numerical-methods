use super::*;

impl<const N: usize> Mat<N, N> {
    /// Apply backward substitution on Ax = b, with A := self.
    pub fn backward_sub(&self, b: &Mat<N, 1>) -> Mat<N, 1> {
        assert!(
            self.is_upper_triangular(),
            "A needs to be upper-triangular:\n{self:?}"
        );
        let mut x = b.clone();
        let n = self.ncols();
        for k in self.col_iter().rev() {
            let mut s = b[k];
            for j in k + 1..n + 1 {
                s -= self[(k, j)] * x[j];
            }
            x[k] = s / self[(k, k)];
        }
        x
    }

    /// Trace: sum of elements on the diagonal
    pub fn trace(&self) -> R {
        (1..=N).map(|i| self[(i, i)]).sum()
    }
}
