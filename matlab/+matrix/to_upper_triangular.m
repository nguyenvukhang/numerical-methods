function A = to_upper_triangular(A)
  n = size(A,1); for i=1:n; for j=1:i-1; A(i,j)=0; end; end
end
