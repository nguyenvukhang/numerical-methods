% let octave know that this is a script file.
% https://docs.octave.org/latest/Script-Files.html
1;

function test_ca32d2b() for _ = 1:10
  A = random.symmetric_positive_definite(5);
  L = matrix.to_lower_triangular(na.cholesky(A));
  assert(L * L', A, 1e-10);
end end

function test_f544868() for _ = 1:10
  A = rand(5, 5);
  [Q, R] = na.gram_schmidt_normalized(A);
  assert(Q * R, A, 1e-10);
  [Q, R] = na.gram_schmidt(A);
  assert(Q * R, A, 1e-10);
end end

function test__horners_method() for _ = 1:10
  p = rand(1, 6); x = rand();
  assert(polyval(p, x), na.horners_method(p, x));
end end

function test__householder()
  A = [1 -4;
       2  3;
       2  2];
  [Q, R] = na.householder(A);
  assert(Q * R, A, 1e-14);
  A = [-2  5;
        3  8;
        1  2];
  [Q, R] = na.householder(A);
  assert(Q * R, A, 1e-14);
  A = [-2  5  7;
        3  8  1;
        1  2  1];
  [Q, R] = na.householder(A);
  assert(Q * R, A, 1e-14);
  A = [-8   5   7  -12;
        3   8   1   12;
       -1   2   1   90;
        0  -8   2    8];
  [Q, R] = na.householder(A);
  assert(Q * R, A, 1e-13);
end

function test__givens()
  A = [1 -4;
       2  3;
       2  2];
  [Q, R] = na.givens_rotations(A);
  assert(Q * R, A, 1e-14);
  A = [-2  5;
        3  8;
        1  2];
  [Q, R] = na.givens_rotations(A);
  assert(Q * R, A, 1e-14);
  A = [-2  5  7;
        3  8  1;
        1  2  1];
  [Q, R] = na.givens_rotations(A);
  assert(Q * R, A, 1e-14);
  A = [-8   5   7  -12;
        3   8   1   12;
       -1   2   1   90;
        0  -8   2    8];
  [Q, R] = na.givens_rotations(A);
  assert(Q * R, A, 1e-13);
end

% Obtain the coefficients of Newton (polynomial) interpolation, given
% the data points in X and Y.
function b = divided_differences(X, Y)
  b = Y; n = size(X, 2);
  for j = 1:n
    for k = n:-1:j+1
      b(k) = (b(k)-b(k-1)) / (X(k)-X(k-j));
    end
  end
end

% x is value to evaluate at.
% b contains the coefficients of Newton interpolation.
% X contains the interpolating nodes.
function v = newtonval(x, b, X)
  nb = 1;   % newton basis function, evaluated at x
  v = b(1); % the initial value is N₀(x)b₀, but N₀(x)=1 always.
  n = size(b, 2);
  for i = 1:n-1
    nb = nb * (x-X(i));  % update the Newton basis function
    v = v + b(i+1) * nb; % build the linear combination incrementally
  end
end

function test__newton_interpolation() for _ = 1:10
  X = rand(5, 1)'; Y = rand(5, 1)'; x = rand();
  matlab_evaluate = polyval(polyfit(X, Y, 4), x);
  newton_evaluate = newtonval(x, divided_differences(X, Y), X);
  assert(matlab_evaluate, newton_evaluate, 1e-6)
end end

test_ca32d2b();
test_f544868();
test__householder();
test__givens();
test__horners_method();
test__newton_interpolation();
disp("[All tests passed!]");
