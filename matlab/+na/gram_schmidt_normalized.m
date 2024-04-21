function [Q, R] = gram_schmidt_normalized(A)
  [m, n] = size(A);
  for i = 1:n
    Q(:,i) = A(:,i);
    for j = 1:i-1
      Q(:,i) = Q(:,i) - vec.project(Q(:,j), A(:,i));
    end
  end
  for i = 1:n
    Q(:,i) = Q(:,i) / norm(Q(:,i));
  end
  for i = 1:n
    for j = i:n
      R(i,j) = dot(Q(:,i),A(:,j));
    end
  end
end
