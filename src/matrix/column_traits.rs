use super::*;

impl<const M: usize> Mat<M, 1> {
    /// Standard dot product.
    pub fn dot(&self, rhs: &Self) -> R {
        let mut v = self[1] * rhs[1];
        (2..self.nrows() + 1).for_each(|i| v += self[i] * rhs[i]);
        v
    }

    /// Orthogonal projection using standard dot product.
    pub fn project(&self, u: &Self) -> Self {
        u.dot(self) / u.dot(u) * u
    }

    /// Euclidean norm.
    pub fn l2_norm(&self) -> R {
        self.dot(self).sqrt()
    }

    /// Converts the column vector into the `k`-th canonical basis
    /// vector. (i.e. the `k`-th column of the identity matrix)
    pub fn canonical_basis(&mut self, k: usize) {
        self.row_iter().for_each(|i| self[i] = if i == k { 1. } else { 0. });
    }

    /// Returns a single vector of values.
    pub fn as_vec(&self) -> Vec<R> {
        self.data[0].into()
    }

    /// Returns true if all elements are unique.
    pub fn has_unique_elements(&self) -> bool {
        for i in 1..=M {
            for j in 1..i {
                if self[i] == self[j] {
                    return false;
                }
            }
        }
        true
    }
}
