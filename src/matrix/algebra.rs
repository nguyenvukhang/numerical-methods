use super::Mat;
use std::ops::{Add, Mul, Sub};

macro_rules! op {
    (add, $a:ty, $b:ty) => {
        impl<const M: usize, const N: usize> Add<$b> for $a {
            type Output = Mat<M, N>;
            fn add(self, rhs: $b) -> Self::Output {
                self.on_each2(&rhs, |a, b| a + b)
            }
        }
    };
    (sub, $a:ty, $b:ty) => {
        impl<const M: usize, const N: usize> Sub<$b> for $a {
            type Output = Mat<M, N>;
            fn sub(self, rhs: $b) -> Self::Output {
                self.on_each2(&rhs, |a, b| a - b)
            }
        }
    };
    (mul, $a:ty, $b:ty) => {
        impl<const M: usize, const N: usize, const P: usize> Mul<$b> for $a {
            type Output = Mat<M, N>;
            fn mul(self, rhs: $b) -> Self::Output {
                let mut m = Mat::<M, N>::new();
                for i in self.row_iter() {
                    for j in rhs.col_iter() {
                        for k in self.col_iter() {
                            m[(i, j)] += self[(i, k)] * rhs[(k, j)]
                        }
                    }
                }
                m
            }
        }
    };
}

macro_rules!x4{
    (mul)=>{op!(mul,Mat<M,P>,Mat<P,N>);op!(mul,&Mat<M,P>,Mat<P,N>);op!(mul,Mat<M,P>,&Mat<P,N>);op!(mul,&Mat<M,P>,&Mat<P,N>);};
    ($x:ident)=>{op!($x,Mat<M,N>,Mat<M,N>);op!($x,&Mat<M,N>,Mat<M,N>);op!($x,Mat<M,N>,&Mat<M,N>);op!($x,&Mat<M,N>,&Mat<M,N>);};
}

x4!(add);
x4!(sub);
x4!(mul);
