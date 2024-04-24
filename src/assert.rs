/// Asserts that two matrices are the same, with an relative tolerance.
#[macro_export]
macro_rules! assert_eq_mat {
    ($l:expr, $r:expr) => {
        assert_eq_mat!($l, $r, 1e-5);
    };
    ($l:expr, $r:expr, $tol:expr) => {
        match (&$l, &$r) {
            (left, right) => {
                assert!(
                    !right.contains_nan(),
                    "Received matrix contains NaN:\nright: {right}",
                );
                assert!(
                    !left.contains_nan(),
                    "Received matrix contains NaN:\nleft: {left}",
                );
                assert!(
                    right.eq(left, $tol),
                    "Matrices do not match:\nleft: {left:?}\nright: {right:?}\ndiff: {:?}", left - right,
                );
            }
        }
    };
}

/// Asserts that two matrices are the same, with an relative tolerance.
#[macro_export]
macro_rules! assert_ne_mat {
    ($l:expr, $r:expr) => {
        assert_ne_mat!($l, $r, 1e-5);
    };
    ($l:expr, $r:expr, $tol:expr) => {
        match (&$l, &$r) {
            (left, right) => {
                assert!(
                    !right.contains_nan(),
                    "Received matrix contains NaN:\nright: {right}",
                );
                assert!(
                    !left.contains_nan(),
                    "Received matrix contains NaN:\nleft: {left}",
                );
                assert!(
                    !right.eq(left, $tol),
                    "Matrices are too close:\nleft: {left:?}\nright: {right:?}\ndiff: {:?}", left - right,
                );
            }
        }
    };
}

/// Assert that two scalars are the same, with an relative tolerance.
#[macro_export]
macro_rules! assert_eq_tol {
    ($l:expr, $r:expr, $tol:expr) => {
        assert!(!$l.is_nan(), "Left value is NaN");
        assert!(!$r.is_nan(), "Right value is NaN");
        assert!(
            $l.rel_diff($r) < $tol,
            "Scalars do not match:\nleft: {}\nright: {}",
            $l,
            $r
        )
    };
}
