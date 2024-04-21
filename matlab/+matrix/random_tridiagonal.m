function A = tridiagonal(n)
  A = to_upper_band(to_lower_band(rand(n, n), 1), 1);
end
