// Numerical Analysis functions

pub mod qr_decomp;

use crate::prelude::*;

/// Clones the matrix, and writes the Cholesky factor into the
/// lower-triangular half of the matrix.
///
/// Input matrix MUST be symmetric positive definite.
pub fn cholesky<const N: usize>(A: &Mat<N, N>) -> Mat<N, N> {
    let mut A = A.clone();
    let n = A.dimensions().0;
    for k in 1..n {
        A[(k, k)] = A[(k, k)].sqrt();
        for j in k + 1..n + 1 {
            A[(j, k)] = A[(j, k)] / A[(k, k)];
        }
        for j in k + 1..n + 1 {
            for i in j..n + 1 {
                A[(i, j)] -= A[(i, k)] * A[(j, k)];
            }
        }
    }
    A[(n, n)] = A[(n, n)].sqrt();
    A
}

#[test]
fn cholesky_test() {
    let mut i = 0;
    while i < REPS {
        let A = symmetric_positive_definite::<5>();
        let L = cholesky(&A).lower_triangular();
        if L.contains_nan() {
            continue;
        }
        let LT = L.transpose();
        assert_eq_mat(A, L * LT, 1e-15);
        i += 1;
    }
}

/// Compute the (reduced) QR factorization of the matrix A via the
/// Gram-Schmidt process. Returns (Q, R) tuple.
pub fn gram_schmidt<const M: usize, const N: usize>(
    A: &Mat<M, N>,
) -> (Mat<M, N>, Mat<N, N>) {
    let mut Q = A.clone();
    for i in A.col_iter() {
        for j in 1..i {
            Q.set_col(i, Q.col(i) - A.col(i).project(&Q.col(j)));
        }
    }

    let mut R = Mat::new();

    // Normalize the columns of Q.
    for j in A.col_iter() {
        let c = Q.col(j);
        Q.set_col(j, &c / c.l2_norm())
    }

    let n = A.ncols();

    for i in A.col_iter() {
        for j in i..n + 1 {
            R[(i, j)] = Q.col(i).dot(&A.col(j))
        }
    }
    (Q, R)
}

#[test]
fn gram_schmidt_test() {
    for _ in 0..REPS {
        let A = Mat::<5, 5>::rand();
        let (Q, R) = gram_schmidt(&A);
        assert_eq_mat(A, &Q * R, 1e-9);

        // check that the columns of Q have norm == 1.
        for j in Q.col_iter() {
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
    let n = A.ncols();
    for k in A.col_iter().rev() {
        let mut s = b[k];
        for j in k + 1..n + 1 {
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
        assert_eq_mat(A * x, b, 1e-8);
    }
}

/// Determine the dominant eigenvector of a matrix, and its
/// corresponding eigenvalue.
pub fn power_iteration<const N: usize>(A: &Mat<N, N>) -> (R, Mat<N, 1>) {
    let mut v = A.col(1);
    loop {
        let mut v2 = A * &v;
        v2.l2_normalize();
        if (&v2 - &v).l2_norm() < 1e-15 {
            v = v2;
            break (v.dot(&(A * &v)), v);
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
        assert_eq_mat(&A * &v, lambda * v, 1e-10);
    }
}

/// Rayleigh Quotient.
/// Useful for calculating the eigenvalue of `v` when it is known that
/// it is an eigenvector of `A`.
pub fn rayleigh_quotient<const N: usize>(v: &Mat<N, 1>, A: &Mat<N, N>) -> R {
    v.dot(&(A * v)) / v.dot(v) // = vᵀAv/vᵀv
}

/// Inverse iteration.
///
/// `a` is the shift.
pub fn inverse_iteration<const N: usize>(
    A: &Mat<N, N>,
    a: R,
) -> (R, Mat<N, 1>) {
    let mut v = A.col(1);
    let B = A - a * eye();

    loop {
        v = B.solve_lls(&v);
        v.l2_normalize();

        if v.is_eigenvector_of(&B, 1e-8) {
            break (v.dot(&(A * &v)), v);
        }
    }
}

#[test]
fn inverse_iteration_test() {
    const N: usize = 6;
    for _ in 0..SMALL_REPS {
        let A = Mat::<N, N>::rand();
        let (lambda, v) = inverse_iteration(&A, A.l1_norm());
        assert_eq_mat(&A * &v, lambda * &v, 1e-8);
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
    let mut lambda = v.dot(&(A * &v));
    let I = eye();

    let mut k = 0;
    const LIMIT: usize = 100;

    loop {
        let B = A - lambda * &I;
        v = B.solve_lls(&v);
        v.l2_normalize();
        lambda = v.dot(&(A * &v));

        if k > LIMIT {
            return Err(Error::NoEigenvalues);
        }
        k += 1;

        println!("λ[{k}]: {lambda}, v: {}", v.t());

        if v.is_eigenvector_of(&B, 1e-8) {
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
    cholesky(&A);
    gram_schmidt(&A);
    horners(&vec![], 1.);
    backward_sub(&A, &b);
    power_iteration(&A);
    rayleigh_quotient(&b, &A);
    inverse_iteration(&A, 0.);
    rayleigh_quotient_iteration(&A);
}
