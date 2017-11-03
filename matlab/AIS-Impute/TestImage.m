clear; clc; close all;
dataset = 'grid';
img = imread(strcat('../noncvx-lowrank(ICDM 2015)/data/images/', dataset, '.jpg'));

if(size(img, 2) > size(img, 1))
    img = permute(img, [2,1,3]);
end

imgSize = size(img);

img = double(img)/255;
imgMean = mean(img(:));
img = img - imgMean;
img = img/std(img(:));

img = reshape(img, imgSize(1), prod(imgSize)/imgSize(1));

missRatio = 0.95;
noisRatio = 0.05;

mask = (rand(size(img)) > missRatio);
G = randn(size(img))*noisRatio;
traD = img + G;
traD = traD.*mask;
traD = sparse(traD);

clear maxk missRatio noisRatio mask G;

para.maxIter = 5000;
para.tol = 1e-6;
para.maxR = 50 ;

%% ---------------------------------------------------------------
mask = (rand(size(img)) > 0.9);
tstD = img.*mask;
tstD = sparse(tstD);

[tstRow, tstCol, tstVal] = find(tstD);

para.test.row  = tstRow;
para.test.col  = tstCol;
para.test.data = tstVal;
para.test.m = size(tstD, 1);
para.test.n = size(tstD, 2);

clear tstRow tstCol tstVal tstD;

%% ---------------------------------------------------------------
lambdaMax = 4;
gridLambda = lambdaMax*(0.8).^(0:9);

gridNMSE = zeros(2, size(gridLambda,2 ));
gridRank = zeros(1, size(gridLambda,2 ));
for g = 1:length(gridLambda) 
    lambda = gridLambda(g);
    
    % [U, S, V] = SoftImpute(traD, lambda, para );
    [U, S, V] = AISImpute(traD, lambda, para );
    gridRank(g) = nnz(S);
    
    X = U*S*V';
    gridNMSE(1, g) = sqrt(norm(X - img, 'fro')^2/numel(img));
    
    [ U, S, V ] = PostProcess(traD, U, V, S);
    X = U*S*V';
    
    gridNMSE(2, g) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
    
    if(g > 1 && gridNMSE(2, g) > gridNMSE(2, g - 1))
        break;
    end
end

gridNMSE = gridNMSE(2,1:g);
[~, lambda] = min(gridNMSE);
gndRank = gridRank(lambda);
para.maxR = ceil(gndRank*1.2);

lambda = gridLambda(lambda);

clear gridNMSE gridRank g X U S V gridLambda;

%% active --------------------------------------------------------
method = 1;
t = tic;
[U, S, V, out{method}] = ActiveSubspace(traD, lambda, para );
% [U, S, V, out{1}]=mc_alt(traData', lambda, para);
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
hold on;
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
hold on;
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('active');

%% boost ---------------------------------------------------------
method = 2;
t = tic;
[U, S, V, out{method}] = Boost( traD, lambda, para);
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
hold on;
plot(out{method}.Time, out{method}.RMSE);
figure(2);
hold on;
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('boost');

%% TR ------------------------------------------------------------
method = 3;
t = tic;
[U, S, V, out{method}] = MMBS( traD, lambda, para );
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('TR');

%% ALT-Impute ----------------------------------------------------
method = 4;
t = tic;
[U, S, V, out{method}] = SoftImputeALS( traD, lambda, para.maxR, para );
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('ALT-Impute');

%% SSGD ----------------------------------------------------------
method = 5;
t = tic;
[U, S, V, out{method}] = SSGD( traD, lambda, gndRank, para );
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('SSGD');

%% LMaFit --------------------------------------------------------

gridNMSE = zeros(1, 10);
for g = 1:10   
    [U, S, V] = FixedRank( traD, g, para );
    
    X = U*S*V';
    gridNMSE(g) = sqrt(norm(X - img, 'fro')^2/numel(img));
    
    if(g > 1 && gridNMSE(g) > gridNMSE(g - 1))
        break;
    end
end

gridNMSE = gridNMSE(1:g);
[~, rnk] = min(gridNMSE);

clear gridNMSE X g U S V;

method = 6;
t = tic;
[U, S, V, out{method}] = FixedRank( traD, rnk, para );
Time(method) = toc(t);

X = U*S*V';
RMSE(method) = out{method}.RMSE(end);
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('LMaFit');

%% APG -----------------------------------------------------------
method = 7;
t = tic;
[U, S, V, out{method}] = APGMatComp( traD, lambda, para );
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('APG');

%% Soft-Impute ---------------------------------------------------
method = 8;
t = tic;
[U, S, V, out{method}] = SoftImpute( traD, lambda, para );
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('Soft-Impute');

%% AIS-Impute ----------------------------------------------------
method = 9;
t = tic;
[U, S, V, out{method}] = AISImpute( traD, lambda, para );
Time(method) = toc(t);

[ U, S, V ] = PostProcess(traD, U, V, S);
X = U*S*V';
RMSE(method) = sqrt(norm(X - img, 'fro')^2/numel(img)); 
clear U S V t;

figure(1);
plot(out{method}.Time, out{method}.RMSE);
figure(2);
semilogy(out{method}.Time, out{method}.obj - min(out{method}.obj));
figure;
X = X + imgMean;
X = reshape(X, imgSize);
imshow(X, []);
title('AIS-Impute');

% clear gndRank img imgMean lambda lambdaMax method rnk traD X mask;
