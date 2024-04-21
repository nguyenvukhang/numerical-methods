function A = positive_definite(n)
  A = rand(n,n) + n*eye(n);
end
