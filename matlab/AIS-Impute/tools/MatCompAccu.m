function [ Accu ] = MatCompAccu( U, V, S, row, col, gndtruth, bias )

U = U*S;
predict = partXY(U', V', row, col, length(gndtruth))';
predict = predict + bias;

predict = sign(predict);

Accu = sum(predict == gndtruth)/length(gndtruth);

end

