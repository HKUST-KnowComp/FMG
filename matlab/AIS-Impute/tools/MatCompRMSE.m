function [ RMSE ] = MatCompRMSE( U, V, S, row, col, gndtruth )

U = U*S;
predict = partXY(U', V', row, col, length(gndtruth))';

predict = predict - gndtruth;
predict = sum(predict.^2);

RMSE = sqrt(predict/length(gndtruth));

end

