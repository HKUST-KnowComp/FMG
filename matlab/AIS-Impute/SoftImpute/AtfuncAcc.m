function [ Y ] = AtfuncAcc( U1, V1, U0, V0, spa, R)

Y = V1*(U1'*R);
Y = Y + V0*(U0'*R);
Y = Y + spa'*R;

end

