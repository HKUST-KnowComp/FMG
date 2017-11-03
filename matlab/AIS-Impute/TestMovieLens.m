clear; clc; close all;

for repeat = 1:5
close all;

dataset = 'movielens10m';
load(strcat('../../noncvx-lowrank(ICDM 2015)/data/recsys/', dataset, '.mat'));
% load('matlab.mat');

clear item user Movie;

[row, col, val] = find(data);
idx = randperm(length(val));

val = val - mean(val);
val = val/std(val);

traIdx = idx(1:floor(length(val)*0.5));
tstIdx = idx(ceil(length(val)*0.5): end);

clear idx;

traData = sparse(row(traIdx), col(traIdx), val(traIdx), size(data,1), size(data,2));

para.maxIter = 100000;
para.tol = 1e-5;

para.test.row  = row(tstIdx);
para.test.col  = col(tstIdx);
para.test.data = val(tstIdx);
para.test.m = size(traData, 1);
para.test.n = size(traData, 2);

% ---------------------------------------------------------------
switch (dataset)
    case 'movielens100k'
        lambdaMax = 30;
        gridLambda = lambdaMax*(0.9).^(0:9);
    case 'movielens1m'
        lambdaMax = 35;
        gridLambda = lambdaMax*(0.95).^(0:9);
    case 'movielens10m'
        gridLambda = 60;
end

clear lambdaMax;

gridRMSE = zeros(2, size(gridLambda,2 ));
gridRank = zeros(1, size(gridLambda,2 ));
for g = 1:length(gridLambda) 
    lambda = gridLambda(g);
    
    [U, S, V] = SoftImpute(traData, lambda, para );
    [U, S, V, out] = AISImpute(traData, lambda, para );
    gridRank(g) = nnz(S);
    
    gridRMSE(1, g) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
    
    [ U, S, V ] = PostProcess(traData, U, V, S);
    
    gridRMSE(2, g) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
    
    if(g > 1 && gridRMSE(2, g) > gridRMSE(2, g - 1))
        break;
    end
end

gridRMSE = gridRMSE(2,1:g);
[~, lambda] = min(gridRMSE);
gndRank = gridRank(lambda);

switch (dataset)
    case 'movielens100k'
        para.maxR = ceil(gndRank*1.2);
    case 'movielens1m'
        para.maxR = ceil(gndRank*1.2);
    case 'movielens10m'
        para.maxR = ceil(gndRank*1.5);
end

lambda = gridLambda(lambda);

clear gridRMSE gridRank g X U S V gridLambda out data;

% active --------------------------------------------------------
method = 1;
t = tic;
[U, S, V, out{method}] = ActiveSubspace(traData, lambda, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));

figure(1);
plot(out{method}.Time, out{method}.RMSE);
hold on;
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
hold on;

% boost ---------------------------------------------------------
method = 2;
t = tic;
[U, S, V, out{method}] = Boost( traData, lambda, para);
Time(method) = toc(t);
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

% % TR ------------------------------------------------------------
% method = 3;
% t = tic;
% [U, S, V, out{method}] = MMBS( traData, lambda, para );
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
% figure(1);
% plot(out{method}.Time, out{method}.RMSE);
% figure(2);
% semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

% ALT-Impute ----------------------------------------------------
method = 4;
t = tic;
[U, S, V, out{method}] = SoftImputeALS( traData, lambda, para.maxR, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

% % SSGD ----------------------------------------------------------
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
% figure(1);
% plot(out{method}.Time, out{method}.RMSE);
% figure(2);
% semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

% LMaFit --------------------------------------------------------
% gridRMSE = zeros(1, 10);
% for g = 1:10   
%     [U, S, V] = FixedRank( traData, g, para );
% 
%     gridRMSE(g) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
%     
% %     if(g > 1 && gridRMSE(g) > gridRMSE(g - 1))
% %         break;
% %     end
% end
% 
% gridRMSE = gridRMSE(1:g);
% [~, rnk] = min(gridRMSE);
% 
% clear gridRMSE X g U S V;

switch (dataset)
    case 'movielens100k'
        rnk = 3;
    case 'movielens1m'
        rnk = 7;
    case 'movielens10m'
        rnk = 12;
end

method = 6;
t = tic;
[U, S, V, out{method}] = FixedRank( traData, rnk, para );
Time(method) = toc(t);

RMSE(2, method) = out{method}.RMSE(end);

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

% %% APG -----------------------------------------------------------
% method = 7;
% t = tic;
% [U, S, V, out{method}] = APGMatComp( traData, lambda, para );
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
  
%% Soft-Impute ---------------------------------------------------
method = 8;
t = tic;
[U, S, V, out{method}] = SoftImpute( traData, lambda, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

%% AISImpute -----------------------------------------------------
method = 9;
t = tic;
[U, S, V, out{method}] = AISImpute( traData, lambda, para );
Time(method) = toc(t);
RMSE(1, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));

[ U, S, V ] = PostProcess(traData, U, V, S);
RMSE(2, method) = MatCompRMSE(U, V, S, row(tstIdx), col(tstIdx), val(tstIdx));
clear U S V t;

out{method}.RMSE = out{method}.RMSE - (RMSE(1, method) - RMSE(2, method));

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));

% clear col data gndRank lambda method para rnk row traData traIdx tstIdx val Age movies;

figure(1);
xlabel('time (sec)');
ylabel('testing RMSE');
legend('active', 'boost', 'TR', 'ALT-Impute', 'SSGD', 'LaMFit', 'APG', 'Soft-Impute', 'AIS-Impute')

figure(2);
xlabel('time (sec)');
ylabel('relative error(obj)');
legend('active', 'boost', 'TR', 'ALT-Impute', 'SSGD', 'LaMFit', 'APG', 'Soft-Impute', 'AIS-Impute')

clear traData lambda gndRank method row S t traIdx tstIdx U V val col rnk para;

save(strcat(dataset, '-', num2str(repeat),'.mat'));

end

