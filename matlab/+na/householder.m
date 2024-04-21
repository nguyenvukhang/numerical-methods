function [Q,R] = householder(A)
  [m, n] = size(A);
  x = zeros(1,n);
  for j = 1:n
    a = A(j:m,j);
    e1 = [1;zeros(m-j,1)];

    % obtain normalized normal vector.
    v = norm(a)*e1-a;
    v = v/norm(v);

    % hard-code the zero'd out column.
    A(j,j)=norm(a);
    A(j+1:m,j) = zeros(m-j,1);

    % on each column (i=j+1:n) of A, apply the HH transform.
    A(j:m,j+1:n) = A(j:m,j+1:n) - 2*v*(v'*A(j:m,j+1:n));

    % save the normal vecs into a list to reconstruct Q.
    V(j:m,j)=v(1:m-j+1);
  end

  % reconstruct Q from all the normal vecs.
  I=eye(m); Q=I;
  for i = 1:n
    v = V(:,i);
    Q = Q*(I-2*v*v');
  end
  R = A;
end
