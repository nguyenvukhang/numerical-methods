#![allow(unused)]

use crate::prelude::{Mat, RealTraits, R};

/// Asserts that two matrices are the same, with an absolute tolerance.
pub fn assert_eq_mat<const M: usize, const N: usize, A>(
    left: A,
    right: A,
    abs_tol: R,
) where
    A: AsRef<Mat<M, N>>,
{
    let (left, right) = (left.as_ref(), right.as_ref());
    assert!(
        !right.contains_nan(),
        "Received matrix contains NaN:\nright: {right:?}"
    );
    assert!(
        !left.contains_nan(),
        "Received matrix contains NaN:\nleft: {left:?}"
    );
    assert!(
        right.eq(left, abs_tol),
        "Matrices do not match:\nleft: {left:?}\nright: {right:?}",
    )
}

/// Assert that two scalars are the same, with an absolute tolerance.
pub fn assert_eq_tol(left: R, right: R, abs_tol: R) {
    assert!(!left.is_nan(), "Left value is NaN");
    assert!(!right.is_nan(), "Right value is NaN");
    assert!(
        left.abs_diff(right) < abs_tol,
        "Scalars do not match:\nleft: {left:?}\nright: {right:?}",
    )
}
