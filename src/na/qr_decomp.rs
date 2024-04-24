use crate::prelude::*;

/// Execute a QR decomposition via Householder reflections.
/// This requires M ≥ N.
pub fn householder<const M: usize, const N: usize>(
    A: &Mat<M, N>,
) -> (Mat<M, M>, Mat<M, N>) {
    let mut R = A.clone();
    let (m, n) = R.dimensions();

    assert!(m >= n, "Householder QR requires nrows ≥ ncols");

    let I = Mat::eye();

    let mut Q = I.clone();

    for j in 1..=N {
        let s = R[(j, j)].signum();

        let mut R_j = R.col(j).clone();

        (1..j).for_each(|x| R_j[x] = 0.); // zero-out entries before j.
        let nom = R_j.l2_norm();

        let u = &R_j + s * nom * I.col(j);

        // Forcefully set the known column
        R[(j, j)] = -s * nom;
        (j + 1..m + 1).for_each(|i| R[(i, j)] = 0.);

        let v = &u / u.l2_norm();

        // Rpply the HH transform on the remaining columns.
        for i in j + 1..n + 1 {
            let x = R.col(i);
            let sub = x - 2. * &v * (v.transpose() * x);
            (j..m + 1).for_each(|k| R[(k, i)] = sub[(k, 1)]);
        }

        Q = Q * (&I - 2. * &v * v.transpose());
    }

    (Q, R)
}

#[test]
fn householder_refl_test() {
    for _ in 0..REPS {
        let A = Mat::<6, 6>::rand();
        let (Q, R) = householder(&A);
        assert_eq_mat(&Q * R, A, 1e-10);
    }
}
