use super::{Mat, R};

/// Initalize a column vector.
#[allow(unused)]
pub fn colv<const S: usize>(val: [R; S]) -> Mat<S, 1> {
    Mat::from([val]).transpose()
}

/// Identity matrix of size `n`
#[allow(unused)]
pub fn eye<const N: usize>() -> Mat<N, N> {
    let mut x = Mat::new();
    x.row_iter().for_each(|i| x[(i, i)] = 1.);
    x
}

#[allow(unused)]
pub fn symmetric<const N: usize>() -> Mat<N, N> {
    let mut x = Mat::rand();
    (&x + x.transpose()) / 2.
}

#[allow(unused)]
pub fn symmetric_positive_definite<const N: usize>() -> Mat<N, N> {
    symmetric() + eye()
}
