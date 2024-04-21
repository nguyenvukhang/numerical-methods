function A = to_lower_triangular(A)
  n = size(A,1); for i=1:n; for j=i+1:n; A(i,j)=0; end; end
end
