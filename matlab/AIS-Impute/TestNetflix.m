clear; clc;

for repeat = 1:5
    
ratio = 10;

dataset = 'netflix';
load(strcat('../../noncvx-lowrank(ICDM 2015)/data/recsys/', dataset, '.mat'));

[row, col, val] = find(data);

val = val - mean(val);
val = val / std(val);

idx = randperm(length(val));

traIdx = idx(1:floor(length(val)*0.5));
tstIdx = idx(ceil(length(val)*0.5): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx), size(data,1), size(data,2));

clear data;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);
para.test.m = size(traData, 1);
para.test.n = size(traData, 2);

lambda = 0.5*sqrt(sum(val.^2));

clear row col val tstIdx traIdx;

para.maxIter = 10000;
para.tol = 1e-6;

switch(ratio)
    case 10
        para.maxR = 5;
    case 20
        para.maxR = 20;
    case 30
        para.maxR = 130;
end

lambda = lambda/ratio;

% %% --------------------------------------------------------------
% method = 1;
% t = tic;
% [U, S, V, out{method}] = ActiveSubspace( traData, lambda, para );
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
% clear U S V t;
% 
% % ALT-Impute ----------------------------------------------------
% method = 2;
% t = tic;
% para.decay = 0.1;
% [U, S, V, out{method}] = SoftImputeALS( traData, lambda, para.maxR, para );
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
% clear U S V t;

% ---------------------------------------------------------------
switch(ratio)
    case 10
        rnk = 3;
    case 20
        rnk = 14;
    case 30
        rnk = 116;
end

method = 3;
t = tic;
para.decay = 0.1;
[U, S, V, out{method}] = FixedRank( traData, rnk, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
clear U S V t rnk;
  
%% --------------------------------------------------------------
method = 4;
t = tic;
[U, S, V, out{method}] = SoftImpute( traData, lambda, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
clear U S V t;

%% --------------------------------------------------------------
method = 5;
t = tic;
[U, S, V, out{method}] = AISImpute( traData, lambda, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, para.test.data);
clear U S V t;

clear row col lambda method para t traData traIdx tstIdx val data;

save(strcat(dataset, '-', num2str(ratio), '-', num2str(repeat),'.mat'));

end
