function [ Y ] = AfuncAcc( U1, V1, U0, V0, spa, R)

Y = U1*(V1'*R);
Y = Y + U0*(V0'*R);
Y = Y + spa*R;

end

