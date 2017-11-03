% clear; clc; close all;
% dataset = 'movielens1m';
% load(strcat('../../noncvx-lowrank(ICDM 2015)/data/recsys/', dataset, '.mat'));
% % load('matlab.mat');
% 
% clear item user;
% 
% [row, col, val] = find(data);
% idx = randperm(length(val));
% 
% val = val - mean(val);
% val = val/std(val);
% 
% traIdx = idx(1:floor(length(val)*0.8));
% tstIdx = idx(ceil(length(val)*0.8): end);
% 
% clear idx;
% 
% traData = sparse(row(traIdx), col(traIdx), val(traIdx), ...
%     size(data,1), size(data,2));
% 
% 
% 
% para.test.row  = row(tstIdx);
% para.test.col  = col(tstIdx);
% para.test.data = val(tstIdx);
% para.test.m = size(traData, 1);
% para.test.n = size(traData, 2);


para.maxIter = 200;
para.tol = 1e-3;
para.maxR = 100;

%% ------------------------------------------------------------------------
method = 1;
[U, S, V, out{method}] = SketchCG(traData, 3250, para);

RMSE(method) = MatCompRMSE(U, V, S, para.test.row, para.test.col, ...
    para.test.data);

% %% ------------------------------------------------------------------------
% method = 2;
% lambda = 30;
% [U, S, V, out{method}] = AISImpute( traData, lambda, para );
% 
% RMSE(method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% clear U S V t;



