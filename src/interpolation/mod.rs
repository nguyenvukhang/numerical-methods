mod lagrange;
mod newton;
mod traits;

pub use lagrange::*;
pub use newton::*;

use crate::prelude::*;

#[allow(unused)]
fn demo() {
    let xs = Mat::from([[-2., 0., 1., 2.]]).t();
    let ys = Mat::from([[-5., 3., 1., 11.]]).t();
    NewtonInterpolation::new(&xs, &ys);
    LagrangeInterpolation::new(&xs, &ys);
}
