% Evaluates polynomial p at value x. Should return the same value as
% `polyval(p, x)`.
%
% `x` is a number
% `p` is a row matrix of coefficients, highest degree first.
function v = horners_method(p, x)
  v = p(1);
  for k = 2:size(p, 2)
    v = p(k) + (x * v);
  end
end
