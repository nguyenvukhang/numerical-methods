use super::{Mat, R};

use std::ops;

pub trait RealTraits {
    fn abs_diff(&self, rhs: Self) -> R;
}

impl RealTraits for R {
    fn abs_diff(&self, rhs: Self) -> R {
        if self > &rhs {
            self - rhs
        } else {
            rhs - self
        }
    }
}

macro_rules! op {
    ($fn:ident, $trait:ident, $op:tt, $v:ty, $s:ty) => {
        impl<const M: usize, const N: usize> ops::$trait<$s> for $v {
            type Output = Mat<M, N>;
            fn $fn(self, rhs: $s) -> Self::Output {
                self.on_each(|v| v $op rhs)
            }
        }
        impl<const M: usize, const N: usize> ops::$trait<$v> for $s {
            type Output = Mat<M, N>;
            fn $fn(self, rhs: $v) -> Self::Output {
                rhs.on_each(|v| v $op self)
            }
        }
    };
}

macro_rules!x4{($x:ident,$t:ident,$o:tt)=>{op!($x,$t,$o,Mat<M,N>,R);op!($x,$t,$o,&Mat<M,N>,R);op!($x,$t,$o,Mat<M,N>,&R);op!($x,$t,$o,&Mat<M,N>,&R);};}

x4!(mul, Mul, *);
x4!(div, Div, /);

/// Convert a 1x1 matrix into a scalar.
impl From<Mat<1, 1>> for R {
    fn from(value: Mat<1, 1>) -> Self {
        value[(1, 1)]
    }
}
