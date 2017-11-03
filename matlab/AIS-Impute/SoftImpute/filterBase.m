function [ R ] = filterBase( V1, V0, tol )

A = [V1, V0];
[U, S, ~] = svd(A'*A);

S = diag(S);
S = S(S > tol);
U = U(:, 1:length(S));
R = A*U;

% R = [V1, V0];
% [R, ~, ~] = svd(R, 'econ');
% R = R(:, 1)

end

