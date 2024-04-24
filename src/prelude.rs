pub(crate) use crate::matrix::*;

#[cfg(test)]
pub(crate) const REPS: usize = 0x1000;

#[cfg(test)]
pub(crate) const SMALL_REPS: usize = 0x100;

pub(crate) type R = f64;

pub(crate) type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
#[allow(unused)]
pub enum Error {
    TooManyIterations(usize),
    NoEigenvalues,
}
