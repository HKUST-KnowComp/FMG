clear; clc; close all;
dataset = 'movielens1m';
load(strcat('../../noncvx-lowrank(ICDM 2015)/data/recsys/', dataset, '.mat'));

clear item user;

[row, col, val] = find(data);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.5));
tstIdx = idx(ceil(length(val)*0.5): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx));
traData(size(data,1), size(data,2)) = 0;

para.maxIter = 1000;
para.tol = 1e-8;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);
para.test.m = size(traData, 1);
para.test.n = size(traData, 2);

lambda = 0;
rnk = 2;

%% ALT-Impute ----------------------------------------------------
method = 1;
para.decay = 0;
[~, ~, ~, out{method}] = SoftImputeALS( traData, lambda, rnk, para );

%% LMaFit --------------------------------------------------------
method = 2;
[~, ~, ~, out{method}] = FixedRank( traData, rnk, para );
out{method}.obj = out{method}.obj/2;

%% BFGS ----------------------------------------------------------
method = 3;
para.maxIter = 1000;
[~,~,~,out{method}] = BFGS( traData, lambda, rnk, para );

idx = (out{method}.RMSE < 1.01);
out{method}.obj = out{method}.obj';
out{method}.obj =  out{method}.obj(idx);
out{method}.Time = out{method}.Time(idx);
out{method}.RMSE = out{method}.RMSE(idx);

clear idx;

close all;
figure;
minObj = min(cat(1, out{1}.obj(end), out{2}.obj(end), out{3}.obj(end)));
semilogy(out{1}.Time, out{1}.obj - minObj);
hold on;
semilogy(out{2}.Time, out{2}.obj - minObj);
semilogy(out{3}.Time, out{3}.obj - minObj);
box on;
legend('AltMin', 'LMaFit', 'BFGS');

figure;
semilogx(out{1}.Time, out{1}.RMSE);
hold on;
semilogx(out{2}.Time, out{2}.RMSE);
semilogx(out{3}.Time, out{3}.RMSE);
box on;

legend('AltMin', 'LMaFit', 'BFGS');
