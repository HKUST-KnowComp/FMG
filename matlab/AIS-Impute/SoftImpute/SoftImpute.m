function [U, S, V, output ] = SoftImpute( D, lambda, para )
% D: sparse observed matrix

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = min(size(D));
end

if(isfield(para, 'decay'))
    decay = para.decay;
else
    decay = 0.925;
end

if(~isfield(para, 'exact'))
    para.exact = 1;
end

objstep = 1;

maxIter = para.maxIter;
tol = para.tol*objstep;

lambdaMax = topksvd(D, 1, 5);

[row, col, data] = find(D);
[m, n] = size(D);

% U = randn(size(D, 1), 1);
% V0 = randn(size(D, 2), 1);
% V1 = V0;
% S = 1;

R = randn(n, 1);
U = powerMethod( D, R, 5, 1e-6);
[~, ~, V0] = svd(U'*D, 'econ');
V1 = V0;

Z = sparse(row, col, data, m, n);
curR = 1;

clear D;

obj = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
for i = 1:maxIter
    ti = tic;
    
    lambdai = abs(lambdaMax - lambda)*(decay^i) + lambda;
    
    % make up sparse term Z = U*V' +spa
    spa = partXY(U', V1', row, col, length(data));
    spa = data - spa';
    setSval(Z, spa, length(spa));
    
    R = filterBase( V1, V0, 1e-5);
    if(para.exact == 1)
        [U, S, V] = exactSVD(U, eye(size(U,2), size(V1,2)) ,V1, Z, curR);
        
        S = diag(S);
        nnzS = sum(S > lambdai);
        U = U(:, 1:nnzS);
        V = V(:, 1:nnzS);
        S = S(1:nnzS);
        S = S - lambdai;
        S = diag(S);
        
        if(curR <= nnzS)
            curR = curR + 5;
        else
            curR = nnzS + 1; 
        end
        
        if(curR > maxR)
            curR = maxR;
        end

        pwIter = inf;
        V0 = V1;
        V1 = V;
        U = U*S;  
    else
        [Q, pwIter] = powerMethodMatComp( U, V1, Z, R, 5, 1e-10);
        hZ = (Q'*U)*V1' + Q'*Z;
        [ U, S, V ] = proximalOperator(hZ, lambdai);
        U = Q*(U*S);
        V0 = V1;
        V1 = V;
        
        curR = size(R, 2);
    end
    
    objVal = lambda*sum(S(:));
    objVal = objVal + (1/2)*sum(spa.^2);
    
    obj(i) = objVal;
    if(i == 1)
        delta = inf;
    else
        delta = (obj(i-1) - obj(i))/obj(i);
    end

    if(i > 1)
        fprintf('iter: %d; obj: %.3d (dif: %.3d); rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
        i, objVal, delta, nnz(S), lambdai, pwIter, curR)
    else
        fprintf('iter: %d; obj: %.3d; rank %d; lambda: %.1f; power(iter %d, rank %d) \n', ...
        i, objVal, nnz(S), lambdai, pwIter, curR)
    end
    
    if(i == 1)
        Time(i) = toc(ti);
    else
        Time(i) = Time(i - 1) + toc(ti);
    end
    
    % testing performance
    if(isfield(para, 'test'))
        tempS = eye(size(U, 2), size(V1, 2));
        if(para.test.m ~= m)
            RMSE(i) = MatCompRMSE(V1, U, tempS, para.test.row, para.test.col, para.test.data);
        else
            RMSE(i) = MatCompRMSE(U, V1, tempS, para.test.row, para.test.col, para.test.data);
        end
        fprintf('RMSE %.2d \n', RMSE(i));
    end
    
    if(i > 1 && abs(delta) < tol)
        break;
    end
end

[U, S, V] = svd(U, 'econ');
V = V1*V;

output.obj = obj(1:i);
output.rank = nnz(S);
output.RMSE = RMSE(1:i);
output.Time = Time(1:i);

end
