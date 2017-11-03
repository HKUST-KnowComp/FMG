function [U,S,V] = exactSVD(U, S, V, Z, k)
%Find the singular values of U S V' + Y
%Y is a sparse matrix

m = size(U,1);
n = size(V,1);
Afunc = @(x) (U*(S*(V'*x)) + Z*x);
Atfunc = @(x) (V*(S*(U'*x)) + Z'*x);
[U, S, V] = lansvd(Afunc,Atfunc, m,n,k,'L');

end