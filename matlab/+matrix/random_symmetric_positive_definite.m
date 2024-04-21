function A = symmetric_positive_definite(n)
  A = symmetric(n) + n*eye(n);
end
