// Numerical Analysis functions

pub mod qr_decomp;

use crate::prelude::*;

/// Clones the matrix, and writes the Cholesky factor into the
/// lower-triangular half of the matrix.
///
/// Input matrix MUST be symmetric positive definite.
pub fn cholesky<const N: usize>(A: &mut Mat<N, N>) {
    for k in 1..N {
        A[(k, k)] = A[(k, k)].sqrt();
        for j in k + 1..=N {
            A[(j, k)] = A[(j, k)] / A[(k, k)];
        }
        for j in k + 1..=N {
            for i in j..=N {
                A[(i, j)] -= A[(i, k)] * A[(j, k)];
            }
        }
    }
    A[(N, N)] = A[(N, N)].sqrt();
}

#[test]
fn cholesky_test() {
    let mut i = 0;
    while i < REPS {
        let A = Mat::<5, 5>::symmetric_positive_definite();
        let mut L = A.clone();
        cholesky(&mut L);
        L.to_lower_triangular();
        if L.contains_nan() {
            continue;
        }
        let LT = L.transpose();
        assert_eq_mat!(A, L * LT, 1e-15);
        i += 1;
    }
}

/// Compute the (reduced) QR factorization of the matrix A via the
/// Gram-Schmidt process. Returns (Q, R) tuple.
pub fn gram_schmidt<const M: usize, const N: usize>(
    A: &Mat<M, N>,
) -> (Mat<M, N>, Mat<N, N>) {
    let mut Q = A.clone();

    // Obtain the orthogonal matrix Q.
    for i in 1..=N {
        for j in 1..i {
            let prj = A.col(i).project(Q.col(j));
            let mut col = Q.col_mut(i);
            col -= prj;
        }
    }

    // Normalize the columns of Q.
    (1..=N).for_each(|j| Q.col_mut(j).l2_normalize());

    // Since A=QR and QᵀQ=I, we obtain R with QᵀA.
    //
    // Since we know that R is upper-triangular, we can skip the computation
    // for the lower-triangular portion.
    let x = |i, j| if i <= j { Q.col(i).dot(A.col(j)) } else { 0. };
    let R = Mat::from_fn(x);

    (Q, R)
}

#[test]
fn gram_schmidt_test() {
    for _ in 0..REPS {
        let A = Mat::<5, 5>::rand();
        let (Q, R) = gram_schmidt(&A);
        assert_eq_mat!(A, &Q * R, 1e-9);

        // check that the columns of Q have norm == 1.
        for j in 1..=5 {
            let norm = Q.col(j).l2_norm();
            assert!(norm.abs_diff(1.) < 1e-9);
        }
    }
}

/// Evaluate a polynomial in linear time, using Horner's Method.
/// `p` is read highest-degree first.
pub fn horners(p: &Vec<R>, x: R) -> R {
    if p.is_empty() {
        return 0.;
    }
    let mut v = p[0];
    for k in 1..p.len() {
        v = p[k] + (x * v)
    }
    v
}

#[test]
fn horners_test() {
    fn polyval(p: &Vec<R>, x: R) -> R {
        p.iter()
            .rev()
            .enumerate()
            .fold(0., |v, (i, p)| v + x.powi(i as i32) * p)
    }
    macro_rules! test {
        ($p:expr, $x:expr) => {
            let p: Vec<R> = $p.to_vec();
            let received = horners(&p, $x);
            let expected = polyval(&p, $x);
            assert!(received.abs_diff(expected) < 1e-10);
        };
    }
    test!([1., 2., 3., 4.], 10.);
    test!([0.2, 2.2, 1.9, 4.1], 2.1);
    test!([0.2, 2.2], 0.1);
    test!([1.2], 91.1);
}

pub fn backward_sub<const N: usize>(A: &Mat<N, N>, b: &Mat<N, 1>) -> Mat<N, 1> {
    assert!(A.is_upper_triangular(), "A needs to be upper-triangular:\n{A:?}");
    let mut x = b.clone();
    for k in (1..=N).rev() {
        let mut s = b[k];
        for j in k + 1..=N {
            s -= A[(k, j)] * x[j];
        }
        x[k] = s / A[(k, k)];
    }
    x
}

#[test]
fn backward_sub_test() {
    const N: usize = 6;
    for _ in 0..REPS {
        let A = Mat::<N, N>::rand().upper_triangular();
        let b = Mat::<N, 1>::rand();
        let x = backward_sub(&A, &b);
        assert_eq_mat!(A * x, b, 1e-8);
    }
}

/// Determine the dominant eigenvector of a matrix, and its
/// corresponding eigenvalue.
pub fn power_iteration<const N: usize>(A: &Mat<N, N>) -> (R, Mat<N, 1>) {
    let mut v = A.col(1).clone();
    loop {
        let mut v2 = A * &v;
        v2.l2_normalize();
        if (&v2 - &v).l2_norm() < 1e-15 {
            v = v2;
            break (v.dot(A * &v), v);
        }
        v = v2;
    }
}

#[test]
fn power_iteration_test() {
    const N: usize = 6;
    for _ in 0..REPS {
        let A = Mat::<N, N>::rand();
        let (lambda, v) = power_iteration(&A);
        assert_eq_mat!(&A * &v, lambda * v, 1e-10);
    }
}

/// Rayleigh Quotient.
/// Useful for calculating the eigenvalue of `v` when it is known that
/// it is an eigenvector of `A`.
pub fn rayleigh_quotient<const N: usize>(v: &Mat<N, 1>, A: &Mat<N, N>) -> R {
    v.dot(A * v) / v.dot(v) // = vᵀAv/vᵀv
}

/// Inverse iteration.
///
/// `a` is the shift.
pub fn inverse_iteration<const N: usize>(
    A: &Mat<N, N>,
    a: R,
) -> (R, Mat<N, 1>) {
    let mut v = A.col(1).clone();
    let B = A - a * Mat::eye();

    loop {
        v = B.solve_lls(&v);
        v.l2_normalize();

        if v.is_eigenvector_of(&B, 1e-8) {
            break (v.dot(A * &v), v);
        }
    }
}

#[test]
fn inverse_iteration_test() {
    const N: usize = 6;
    for _ in 0..SMALL_REPS {
        let A = Mat::<N, N>::rand();
        let (lambda, v) = inverse_iteration(&A, A.l1_norm());
        assert_eq_mat!(A * &v, lambda * &v, 1e-8);
    }
}

#[test]
fn inverse_iteration_against_power_iteration() {
    const N: usize = 6;
    for _ in 0..SMALL_REPS {
        let A = Mat::<N, N>::rand();
        let (_, v) = inverse_iteration(&A, A.l1_norm());
        let (_, pv) = power_iteration(&A);
        assert!(pv.eq(&v, 1e-8) || pv.eq(-v, 1e-8));
    }
}

/// Rayleigh Quotient Iteration.
///
/// Currently runs forever on matrices that have no eigenvalues.
pub fn rayleigh_quotient_iteration<const N: usize>(
    A: &Mat<N, N>,
) -> Result<(R, Mat<N, 1>)> {
    let mut v = Mat::rand();
    v.l2_normalize();
    let mut lambda = v.dot(A * &v);

    let mut k = 0;
    const LIMIT: usize = 100;

    loop {
        let mut B = A.clone();
        B.add_identity(-lambda);
        v = B.solve_lls(&v);
        v.l2_normalize();
        lambda = v.dot(A * &v);

        if k > LIMIT {
            return Err(Error::NoEigenvalues);
        }
        k += 1;

        if v.is_eigenvector_of(&B, 1e-9) {
            break Ok((lambda, v));
        }
    }
}

#[test]
fn rayleigh_quotient_iteration_test() {
    const N: usize = 6;
    let mut k = 0;
    while k < SMALL_REPS {
        let A = Mat::<N, N>::rand();
        if let Ok((lambda, v)) = rayleigh_quotient_iteration(&A) {
            assert_eq_mat!(&A * &v, lambda * &v, 1e-8);
            k += 1;
        }
    }
}

/// The allow_unused sink.
#[allow(unused)]
fn demo() {
    let A = Mat::<5, 5>::rand();
    let b = Mat::<5, 1>::rand();
    cholesky(&mut Mat::<1, 1>::zero());
    gram_schmidt(&A);
    horners(&vec![], 1.);
    backward_sub(&A, &b);
    power_iteration(&A);
    rayleigh_quotient(&b, &A);
    inverse_iteration(&A, 0.);
    rayleigh_quotient_iteration(&A);
}
