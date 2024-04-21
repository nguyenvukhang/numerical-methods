use crate::prelude::*;

pub trait Interpolator<const N: usize> {
    /// Evaluates the `k`-th basis vector at `x`.
    fn basis_fn_eval(&self, k: usize, x: R) -> R;

    /// Gets the `k`-th coefficient in the interpolating polynomial.
    fn coeff(&self, k: usize) -> R;

    /// estimate the value of f(x) using known data points.
    fn estimate(&self, x: R) -> R {
        (1..=N).map(|i| self.coeff(i) * self.basis_fn_eval(i, x)).sum()
    }
}
