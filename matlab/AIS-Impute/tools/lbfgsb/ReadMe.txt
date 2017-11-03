If you use linux or Mac, simply compile it by either running "make" in matlab 
(which runs make.m) or by running "make all" in bash (which uses Makefile).  


If you use Windows, The folder contains a lbfgsb.mexw32 which 
was compiled on Win7.
First try if lbfgsb.mexw32 already works (ie simply running it will 
give an error of insufficient input).  
If it doesn't work, switch to lbfgsb_WinXP (ie rename it to lbfgsb.mexw32).
If it still doesn't work, contact me.

Two examples of its usage: test.m and examplehs038.m

More details on the paramters are at parameters.txt.

There are two ways to invoke the lbfgsb solver:
1: use function pointer
[x, f, iter, numCall, flag] = lbfgsb(x0,lb,ub,@computeObjAndGrad,[],@genericcallback, param);

In this case, one must provide a function which 
returns both the function value and the gradient.


2: use string as function name.
[x, f, iter, numCall, flag] = lbfgsb(x0,lb,ub,'computeObjectiveHS038','computeGradientHS038',...
           [],'genericcallback', param);

In this case, one must provide two functions.  
One returns the function value and one returns the gradient.

'flag' encodes the termination reason (in integer).
See parameters.txt for its meaning.