% Test the "lbfgsb" Matlab interface on the Hock & Schittkowski test problem
% #38. See: Willi Hock and Klaus Schittkowski. (1981) Test Examples for
% Nonlinear Programming Codes. Lecture Notes in Economics and Mathematical
% Systems Vol. 187, Springer-Verlag.

% The starting point.
function [] = examples038 ()

x0  = [-3  -1  -3  -1];   % The starting point.
lb  = [-10 -10 -10 -10];  % Lower bound on the variables.
ub  = [+10 +10 +10 +10];  % Upper bound on the variables.

function [f, g] = computeObjAndGrad(x)
    f = computeObjectiveHS038(x);
    g = computeGradientHS038(x);
end

param = [];
param.maxIter = 80;     % max number of iterations
param.maxFnCall = 100;  % max number of calling the function
param.relCha = 1e-5;      % tolerance of constraint satisfaction
param.tolPG = 1e-5;   % final objective function accuracy parameter
param.m = 4;

% Method 1: use function pointer
% If use function pointer, then must use a function which returns
%   both the function value and the gradient
[x, f, iter, numCall, flag] = lbfgsb(x0,lb,ub,@computeObjAndGrad,[],@genericcallback, param);

% Method 2: use string as function name
% If use the function name, then must use two functions.  
% One returns the function value and one returns the gradient.
[x, f, iter, numCall, flag] = lbfgsb(x0,lb,ub,'computeObjectiveHS038','computeGradientHS038',...
           [],'genericcallback', param);

end