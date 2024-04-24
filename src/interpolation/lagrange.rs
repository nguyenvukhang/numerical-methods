use super::traits::Interpolator;
use crate::prelude::*;

pub struct LagrangeInterpolation<const N: usize> {
    xs: Mat<N, 1>,
    ys: Mat<N, 1>,
}

impl<const N: usize> LagrangeInterpolation<N> {
    /// Initialize a new Lagrange interpolator.
    pub fn new(xs: &Mat<N, 1>, ys: &Mat<N, 1>) -> Self {
        assert!(
            xs.has_unique_elements(),
            "Can't do interpolation with non-unique x-values"
        );
        Self { xs: xs.clone(), ys: ys.clone() }
    }
}

impl<const N: usize> Interpolator<N> for LagrangeInterpolation<N> {
    /// Evaluate L_k(x).
    fn basis_fn_eval(&self, k: usize, x: R) -> R {
        let mut v = 1.;
        for j in 1..=N {
            if j != k {
                if x == self.xs[j] {
                    return 0.;
                }
                v *= (x - self.xs[j]) / (self.xs[k] - self.xs[j])
            }
        }
        v
    }

    fn coeff(&self, k: usize) -> R {
        self.ys[k]
    }
}

#[test]
fn lagrange_interpolation_test() {
    use crate::na::horners;

    let xs = Mat::from([[-2., 0., 1., 2.]]).t();
    let ys = Mat::from([[-5., 3., 1., 11.]]).t();
    let lg = LagrangeInterpolation::new(&xs, &ys);
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
