% vector projection of a → u
function x = project(u, a)
  x = dot(u,a) / dot(u,u) * u;
end
