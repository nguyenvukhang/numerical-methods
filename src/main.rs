#![allow(non_snake_case)] // matrices are traditionally written in upper case.

#[macro_use]
mod assert;
mod interpolation;
mod matrix;
mod na;
mod prelude;

use prelude::*;

#[allow(unused)]
fn demo() {
    let A = Mat::<5, 5>::rand();
    let b = Mat::<5, 1>::rand();
    na::cholesky(&mut Mat::<1, 1>::zero());
    na::horners(&vec![], 1.);
    na::backward_sub(&A, &b);
    na::power_iteration(&A);
    na::rayleigh_quotient(&b, &A);
    na::inverse_iteration(&A, 0.);
    na::rayleigh_quotient_iteration(&A);
    na::qr_decomp::gram_schmidt(&A);
}

fn main() -> Result<()> {
    Ok(())
}
