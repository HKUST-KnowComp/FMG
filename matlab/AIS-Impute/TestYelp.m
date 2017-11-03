clear; clc; close all;
dataset = 'amazon';
load(strcat('../../noncvx-lowrank(ICDM 2015)/data/recsys/', dataset, '.mat'));
% load('matlab.mat');

clear item user;

[row, col, val] = find(data);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.8));
tstIdx = idx(ceil(length(val)*0.8): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx));
traData(size(data,1), size(data,2)) = 0;

para.maxIter = 100;
para.tol = 1e-5;
para.decay = 0.01;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);
para.test.m = size(traData, 1);
para.test.n = size(traData, 2);

%% start testing
method = 1;
t = tic;
[U, S, V, out{method}] = SoftImputeALS( traData, 0.0, 10, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

