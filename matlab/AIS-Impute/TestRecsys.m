clear; clc; close all;
dataset = 'movielens10m';
load(strcat('../../noncvx-lowrank(ICDM 2015)/data/recsys/', dataset, '.mat'));
% load('matlab.mat');

clear item user;

[row, col, val] = find(data);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.5));
tstIdx = idx(ceil(length(val)*0.5): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx), ...
    size(data,1), size(data,2));

para.maxIter = 100;
para.tol = 1e-5;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);
para.test.m = size(traData, 1);
para.test.n = size(traData, 2);

%% start testing
switch (dataset)
    case 'movielens100k'
        lambda = 9;
        para.tol = 1e-6;
        
        para.maxR = 100;
    case 'movielens1m'
        lambda = 18.5;
        para.tol = 1e-2;
        
        para.exact = 1;
        para.maxR = 150;
    case 'movielens10m'
        lambda = 45;
        
        para.tol = 1e-5;
        para.maxR = 300;
end

% %% ---------------------------------------------------------------
% switch (dataset)
%     case 'movielens100k'
%         lambdaMax = 30;
%     case 'movielens1m'
%         lambdaMax = 45;
%     case 'movielens10m'
%         lambdaMax = 45;
% end
% 
% gridLambda = lambdaMax*(0.9).^(0:9);
% 
% clear lambdaMax;
% 
% gridRMSE = zeros(2, size(gridLambda,2 ));
% gridRank = zeros(1, size(gridLambda,2 ));
% for g = 1:length(gridLambda) 
%     lambda = gridLambda(g);
%     
%     % [U, S, V] = SoftImpute(traData, lambda, para );
%     [U, S, V] = AISImpute(traData, lambda, para );
%     gridRank(g) = nnz(S);
%     
%     gridRMSE(1, g) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
%     
%     [ U, S, V ] = PostProcess(traData, U, V, S);
%     
%     gridRMSE(2, g) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
%     
%     if(g > 1 && gridRMSE(2, g) > gridRMSE(2, g - 1))
%         break;
%     end
% end
% 
% gridRMSE = gridRMSE(2,1:g);
% [~, lambda] = min(gridRMSE);
% gndRank = gridRank(lambda);
% para.maxR = ceil(gndRank*1.2);
% 
% lambda = gridLambda(lambda);
% 
% clear gridRMSE gridRank g X U S V gridLambda;
% 
% %% active --------------------------------------------------------
% method = 1;
% t = tic;
% [U, S, V, out{method}] = ActiveSubspace(traData, lambda, para );
% % [U, S, V, out{1}]=mc_alt(traData', lambda, para);
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% figure(1);
% plot(out{method}.Time, out{method}.RMSE);
% hold on;
% figure(2);
% semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% hold on;
% 
% %% boost ---------------------------------------------------------
% method = 2;
% t = tic;
% [U, S, V, out{method}] = Boost( traData, lambda, para);
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% figure(1);
% plot(out{method}.Time, out{method}.RMSE);
% figure(2);
% semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% %% TR ------------------------------------------------------------
% method = 3;
% t = tic;
% [U, S, V, out{method}] = MMBS( traData, lambda, para );
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% figure(1);
% plot(out{method}.Time, out{method}.RMSE);
% figure(2);
% semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% %% ALT-Impute ----------------------------------------------------
% method = 4;
% t = tic;
% [U, S, V, out{method}] = SoftImputeALS( traData, lambda, para.maxR, para );
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% figure(1);
% plot(out{method}.Time, out{method}.RMSE);
% figure(2);
% semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% %% SSGD ----------------------------------------------------------
% method = 5;
% t = tic;
% [U, S, V, out{method}] = SSGD( traData, lambda, gndRank, para );
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% % figure(1);
% % plot(out{method}.Time, out{method}.RMSE);
% % figure(2);
% % semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% %% LMaFit --------------------------------------------------------
% gridRMSE = zeros(1, 10);
% for g = 1:10   
%     [U, S, V] = FixedRank( traData, g, para );
% 
%     gridRMSE(g) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
%     
%     if(g > 1 && gridRMSE(g) > gridRMSE(g - 1))
%         break;
%     end
% end
% 
% gridRMSE = gridRMSE(1:g);
% [~, rnk] = min(gridRMSE);
% 
% clear gridNMSE X g U S V;
% 
% method = 6;
% t = tic;
% [U, S, V, out{method}] = FixedRank( traData, rnk, para );
% Time(method) = toc(t);
% 
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% % figure(1);
% % plot(out{method}.Time, out{method}.RMSE);
% % figure(2);
% % semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% %% APG -----------------------------------------------------------
% method = 7;
% t = tic;
% [U, S, V, out{method}] = APGMatComp( traData, lambda, para );
% Time(method) = toc(t);
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% % figure(1);
% % plot(out{method}.Time, out{method}.RMSE);
% % figure(2);
% % semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
%   
% %% Soft-Impute ---------------------------------------------------
% method = 8;
% t = tic;
% [U, S, V, out{method}] = SoftImpute( traData, lambda, para );
% Time(method) = toc(t);
% Time(method) = toc(t);
% RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% 
% [ U, S, V ] = PostProcess(traData, U, V, S);
% RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
% clear U S V t;
% 
% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% % figure(1);
% % plot(out{method}.Time, out{method}.RMSE);
% % figure(2);
% % semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% %% ---------------------------------------------------------------
method = 9;
t = tic;
[U, S, V, out{method}] = AISImpute( traData, lambda, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

% out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));
% 
% % figure(1);
% % plot(out{method}.Time, out{method}.RMSE);
% % figure(2);
% % semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
% 
% para.maxIter = 200;
% [~,~,~,out] = BFGS( traData, 0.001, 7, para );
% 
% idx = (out.RMSE < 1.01);
% out.obj =  out.obj(idx);
% out.Time = out.Time(idx);
% out.RMSE = out.RMSE(idx);
% 
% clear idx;
% 
% % clear col data gndRank lambda method para rnk row traData traIdx tstIdx val;