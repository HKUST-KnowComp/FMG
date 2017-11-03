function [U, S, V, nnzS] = AISSVT( U, S, V, lambda )

S = diag(S);
nnzS = sum(S > lambda);
U = U(:, 1:nnzS);
V = V(:, 1:nnzS);
S = S(1:nnzS);
S = S - lambda;
S = diag(S);

end