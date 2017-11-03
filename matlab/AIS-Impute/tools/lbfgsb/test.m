function [] = test()


x0 = [3; 2];    % Inital value
bl = [-2; 0];   % lower bounds
bu = [inf; 3];  % upper bounds

param = [];
param.maxIter = 10;     % max number of iterations
param.maxFnCall = 100;  % max number of calling the function
param.relCha = 1e-5;      % tolerance of constraint satisfaction
param.tolPG = 1e-5;   % final objective function accuracy parameter
param.m = 10;


function [f, g] = toy_func(x)
    cen = [-1; 1];
    f = 0.5 * norm(x - cen)^2;
    g = x - cen;
end


% At present, only box constraints are implemented in the mex.
[x, f, iter, numCall, flag] = lbfgsb(x0, bl, bu, @toy_func, [], @genericcallback, param) %





end
