function [Q,R] = givens_rotations(A)
  [m, n] = size(A);
  Q = eye(m); R = A;
  for j = 1:n
    for i = j+1:m
      % build the Givens rotation matrix
      d = sqrt(R(j,j)^2 + R(i,j)^2);
      c = R(j,j)/d; s = R(i,j)/d; G = eye(m);
      G(i,i) = c;  G(i,j) = s;
      G(j,i) = -s; G(j,j) = c;
      % Use Givens to use Ajj annihilate Aij
      R = G * R;
      % invert the Givens matrix
      G(i,j) = -G(i,j); G(j,i) = -G(j,i);
      Q = Q * G;
    end
  end
end
