function [Q, maxIter] = powerMethodAccMatComp( U1, V1, U0, V0, spa, bi, R, maxIter, tol)

if(size(U1, 1) > size(V1, 1))
    V1 = V1*(1 + bi);
else
    U1 = U1*(1 + bi);
end

if(size(U0, 1) > size(V0, 1))
    V0 = V0*(-bi);
else
    U0 = U0*(-bi);
end

Y = AfuncAcc( U1, V1, U0, V0, spa, R);

[Q, ~] = qr(Y, 0);
err = zeros(maxIter, 1);
for i = 1:maxIter
    Y = AtfuncAcc( U1, V1, U0, V0, spa, Q);
    Y = AfuncAcc( U1, V1, U0, V0, spa, Y);
    
    [iQ, ~] = qr(Y, 0);
    
    err(i) = norm(iQ(:,1) - Q(:,1), 2);
    Q = iQ;
    
    if(err(i) < tol)
        break;
    end
end

maxIter = i;

end
