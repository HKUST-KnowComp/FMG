function [ s ] = topksvd( X, k, maxIter )

N = size(X, 2);
R = randn(N, k);

if(~exist('maxIter', 'var'))
    maxIter = 5;
end

Q = powerMethod( X, R, maxIter);

X = Q'*X;
s = svd(X);

end

