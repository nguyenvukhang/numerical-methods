use crate::prelude::*;

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

/// Execute a QR decomposition via Householder reflections.
/// This requires M ≥ N.
pub fn householder<const M: usize, const N: usize>(
    A: &Mat<M, N>,
) -> (Mat<M, M>, Mat<M, N>) {
    let mut R = A.clone();

    assert!(M >= N, "Householder QR requires nrows ≥ ncols");

    let I = Mat::eye();

    let mut Q = I.clone();

    for j in 1..=N {
        let s = R[(j, j)].signum();

        let mut v = R.col(j).clone();

        (1..j).for_each(|x| v[x] = 0.); // zero-out entries before j.

        // Forcefully set the known column
        R[(j, j)] = -s * v.l2_norm();

        (j + 1..=M).for_each(|i| R[(i, j)] = 0.);

        // at Rjj times the j-th canonical basis vec to u.
        v[j] -= R[(j, j)];

        v.l2_normalize();
        let vt = v.t();

        // Rpply the HH transform on the remaining columns.
        for i in j + 1..=N {
            let x = R.col(i);
            let sub = x - 2. * &v * (&vt * x);
            (j..=M).for_each(|k| R[(k, i)] = sub[k]);
        }

        // Q = Q * (I - 2vvᵀ);
        let B = &Q * 2. * v * vt;
        Q = Q - B;
    }

    (Q, R)
}

#[test]
fn householder_refl_test() {
    for _ in 0..REPS {
        let A = Mat::<6, 6>::rand();
        let (Q, R) = householder(&A);
        assert_eq_mat!(&Q * R, A, 1e-10);
    }
}
