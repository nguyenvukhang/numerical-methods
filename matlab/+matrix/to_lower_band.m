% Zero a matrix until it has an lower bandwidth of `q`
function A = to_lower_band(A, q)
  n = size(A, 1); for j=1:n; for i=j+q+1:n; A(i,j)=0; end; end
end
