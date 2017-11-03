function [U, S, V] = accExactSVD( U1, V1, U0, V0, spa, bi, k)

m = size(U1,1);
n = size(V1,1);
Afunc  = @(x) (spa*x + (1+bi)*(U1*(V1'*x)) - bi*(U0*(V0'*x)));
Atfunc = @(y) (spa'*y + (1+bi)*(V1*(U1'*y)) - bi*(V0*(U0'*y)));

OPTIONS.tol = 4*eps;
OPTIONS.elr = 1;

rnk = min(min(m,n), ceil(1.5*k));
[U, S, V] = lansvd(Afunc, Atfunc, m, n, rnk, 'L', OPTIONS);

U = U(:, 1:k);
V = V(:, 1:k);

S = diag(S);
S = S(1:k);
S = diag(S);

end