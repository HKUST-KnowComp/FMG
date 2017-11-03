function [Q, maxIter] = powerMethodMatComp( U, V, spa, R, maxIter, tol)
% low rank (U*V') + sparse term (data)

if(isempty(R))
    maxIter = 0;
    Q = randn(size(U,1), 1);
    return;
end

Y = (R'*spa')';
Y = U*(V'*R) + Y;
[Q, ~] = qr(Y, 0);
err = zeros(maxIter, 1);
for i = 1:maxIter
    % Y = A*(A'*Q);
    AtQ = (Q'*spa)';
    AtQ = V*(U'*Q) + AtQ;
    Y = (AtQ'*spa')';
    Y = U*(V'*AtQ) + Y;
    [iQ, ~] = qr(Y, 0);
    
    err(i) = norm(iQ(:,1) - Q(:,1), 2);
    Q = iQ;
    
    if(err(i) < tol)
        break;
    end
end

maxIter = i;

end


