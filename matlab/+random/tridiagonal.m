function A = tridiagonal(n)
  A = matrix.to_upper_band(matrix.to_lower_band(rand(n, n), 1), 1);
end
