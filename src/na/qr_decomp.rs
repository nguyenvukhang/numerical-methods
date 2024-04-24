use crate::prelude::*;

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
        let nom = v.l2_norm();
        let s_nom = s * nom;

        // at s_nom times the j-th canonical basis vec to u.
        v[j] += s_nom;

        // Forcefully set the known column
        R[(j, j)] = -s_nom;
        (j + 1..=M).for_each(|i| R[(i, j)] = 0.);

        v.l2_normalize();
        let vt = v.t();

        // Rpply the HH transform on the remaining columns.
        for i in j + 1..=N {
            let x = R.col(i);
            let sub = x - 2. * &v * (&vt * x);
            (j..=M).for_each(|k| R[(k, i)] = sub[(k, 1)]);
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
