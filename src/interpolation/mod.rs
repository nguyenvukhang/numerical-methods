mod lagrange;
mod newton;
mod traits;

use lagrange::*;
use newton::*;
use traits::*;

use crate::matrix::*;

#[allow(unused)]
fn demo() {
    let xs = Mat::from([[-2., 0., 1., 2.]]).t();
    let ys = Mat::from([[-5., 3., 1., 11.]]).t();
    NewtonInterpolation::new(&xs, &ys);
    LagrangeInterpolation::new(&xs, &ys);
}

pub fn run() {}
