use super::traits::Interpolator;
use crate::prelude::*;

pub struct NewtonInterpolation<const N: usize> {
    xs: Mat<N, 1>,
    coeffs: Mat<N, 1>,
}

fn divided_differences<const N: usize>(
    x: &Mat<N, 1>,
    y: &Mat<N, 1>,
) -> Mat<N, 1> {
    let mut b = y.clone();
    for j in 1..=N {
        for k in (j + 1..=N).rev() {
            b[k] = (b[k] - b[k - 1]) / (x[k] - x[k - j]);
        }
    }
    b
}

impl<const N: usize> NewtonInterpolation<N> {
    /// Initialize a new Lagrange interpolator.
    pub fn new(xs: &Mat<N, 1>, ys: &Mat<N, 1>) -> Self {
        assert!(
            xs.has_unique_elements(),
            "Can't do interpolation with non-unique x-values"
        );
        Self { xs: xs.clone(), coeffs: divided_differences(xs, ys) }
    }
}

impl<const N: usize> Interpolator<N> for NewtonInterpolation<N> {
    /// Evaluate N_k(x).
    fn basis_fn_eval(&self, k: usize, x: R) -> R {
        (1..k).fold(1., |v, j| v * (x - self.xs[j]))
    }

    fn coeff(&self, k: usize) -> R {
        self.coeffs[k]
    }
}

#[test]
fn newton_interpolation_test() {
    use crate::na::horners;

    let xs = Mat::from([[-2., 0., 1., 2.]]).t();
    let ys = Mat::from([[-5., 3., 1., 11.]]).t();
    let lg = NewtonInterpolation::new(&xs, &ys);
    let pl = vec![2., 0., -4., 3.];
    let pl = |x| horners(&pl, x);

    for _ in 0..REPS {
        let rand = Mat::<6, 1>::rand();
        for i in 1..=6 {
            let x = rand[i];
            assert_eq_tol!(pl(x), lg.estimate(x), 1e-4);
        }
    }
}
