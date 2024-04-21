% Zero a matrix until it has an upper bandwidth of `p`
function A = to_upper_band(A, p)
  n = size(A, 1); for i=1:n; for j=i+p+1:n; A(i,j)=0; end; end
end
