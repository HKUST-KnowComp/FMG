function [ U, S, V ] = SubspaceIter( U1, V1, U0, V0, spa, bi, R, maxIter )

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

V = R;
for i = 1:maxIter
    U = AfuncAcc( U1, V1, U0, V0, spa, V);
    [U, ~] = qr(U, 0);
    
    V = AtfuncAcc( U1, V1, U0, V0, spa, U);
    [V, ~] = qr(V, 0);
end

S = AfuncAcc( U1, V1, U0, V0, spa, V);
S = U'*S;

[Us, S, Vs] = svd(S, 'econ');

U = U*Us;
V = V*Vs;

end

