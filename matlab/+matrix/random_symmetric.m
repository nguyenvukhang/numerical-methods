function A = symmetric(n)
  A = rand(n,n); A = (A + A')/2;
end
