function A = symmetric_positive_definite(n)
  A = random.symmetric(n) + n*eye(n);
end
