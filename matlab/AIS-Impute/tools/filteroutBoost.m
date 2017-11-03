function [U, S, V] = filteroutBoost(U, V, maxR)

[QU, RU] = qr(U, 0);
[QV, RV] = qr(V, 0);

R = RU*RV';
[P, S, Q] = svd(R, 'econ');

S = diag(S);
nnzS = min(sum(S > 1e-10), maxR);

P = P(:, 1:nnzS);
Q = Q(:, 1:nnzS);
S = S(1:nnzS);

S = diag(S);
U = QU*P;
V = QV*Q;

end