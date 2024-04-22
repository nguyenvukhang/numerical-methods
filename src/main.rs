#![allow(non_snake_case)] // matrices are traditionally written in upper case.
mod assert;
mod interpolation;
mod matrix;
mod na;
mod prelude;

use prelude::*;

fn main() {
    interpolation::run();
    const N: usize = 5;
    let A = Mat::<N, N>::rand();
    let (lambda, v) =
        na::inverse_iteration(&A, na::rayleigh_quotient(&A.col(1), &A)).unwrap();
}
