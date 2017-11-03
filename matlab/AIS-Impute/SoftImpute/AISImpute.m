function [U0, S, V, output ] = AISImpute( D, lambda, para )
% D: sparse observed matrix (m <= n)

% if(isfield(para, 'decay'))
%     decay = para.decay;
% else
%     decay = 0.75;
% end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

if(isfield(para, 'exact'))
    exact = para.exact;
else
    exact = 0;
end

maxIter = para.maxIter;
tol = para.tol;

% lambdaMax = topksvd(D, 1, 5);

[row, col, data] = find(D);
[m, n] = size(D);

% U = randn(size(D, 1), 1);
% V0 = randn(size(D, 2), 1);
% V1 = V0;
% S = 1;

R = randn(n, 1);
U0 = powerMethod( D, R, 5, 1e-6);
U1 = U0;

[~, ~, V0] = svd(U0'*D, 'econ');
V1 = V0;

a0 = 1;
a1 = 1;

spa = sparse(row, col, data, m, n);
part0 = partXY_blas(U0', V0', row, col, length(data));
part1 = partXY_blas(U1', V1', row, col, length(data));

clear D;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RankIn = zeros(maxIter, 1);
RankOut = zeros(maxIter, 1);

for i = 1:maxIter
    tt = tic;
    
    bi = (a0 - 1)/a1;
    stepsize = i / (i + 5);
    
    % make up sparse term Z = U*V' + spa (blas is faster)
    part0 = data - (1 + bi)*part1' + bi*part0';
    setSval(spa, part0/stepsize, length(part0));
    
    R = filterBase( V1, V0, 1e-5);
    R = R(:, 1:min([size(R,2), maxR]));
    RankIn(i) = size(R, 2);
    
    if(exact == 1)
        [Ui, S, Vi] = accExactSVD( U1, V1, U0, V0, spa, bi, size(R, 2));
        [Ui, S, Vi] = AISSVT(Ui, S, Vi, lambda/stepsize);
        Ui = Ui*S;

        pwIter = inf;
    else
%         [Q, pwIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, R, 10, pwTol);
%         hZ = (1+bi)*(Q'*U1)*V1' - bi*(Q'*U0)*V0' + Q'*spa;
%         
%         [Ui, S, Vi] = svd(hZ, 'econ');
%         Ui = Q*(Ui*S);
        
        pwIter = 3;
        [ Ui, S, Vi ] = SubspaceIter( U1, V1, U0, V0, spa, bi, R, pwIter );
        
        [Ui, S, Vi] = AISSVT(Ui, S, Vi, lambda/stepsize); 
        Ui = Ui*S;
    end

    U0 = U1;
    U1 = Ui;

    V0 = V1;
    V1 = Vi;
    
    RankOut(i) = nnz(S);
    ai = (1 + sqrt(1 + 4*a0^2))/2;
    a0 = a1;
    a1 = ai;
    
    % objective value
    part0 = part1;
    part1 = partXY_blas(Ui', Vi', row, col, length(data));

    objVal = (1/2)*sum((data - part1').^2);
    objVal = objVal + lambda*sum(S(:));
    obj(i) = objVal;

    if(i > 1)
        delta = (obj(i - 1)- objVal)/objVal;
        fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
            i, objVal, delta, nnz(S), lambda, pwIter, size(R, 2) )

        if(delta < 0)
            a0 = 1;
            a1 = 1;
        end
    else
        fprintf('iter: %d; obj: %d; rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
            i, objVal, nnz(S), lambda, pwIter, size(R, 2) )
    end
    
    ti = toc(tt);
    if(i == 1)
        Time(i) = ti;
    else
        Time(i) = Time(i - 1) + ti;
    end

    % testing performance
    if(isfield(para, 'test'))        
        tempS = eye(size(U1, 2), size(V1, 2));
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V1, U1, tempS, ...
                para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U1, V1, tempS, ...
                para.test.row, para.test.col, para.test.data);
        end
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
end

[U0, S, V] = svd(U1, 'econ');
V = V1*V;

output.obj = obj(1:i);
output.Rank = nnz(S);
output.RMSE = RMSE(1:i);
output.RankIn = RankIn(1:i);
output.RankOut = RankOut(1:i);
output.Time = Time(1:i);

end

