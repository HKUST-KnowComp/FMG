#include "matlabexception.h"
#include "matlabscalar.h"
#include "matlabstring.h"
#include "matlabmatrix.h"
#include "arrayofmatrices.h"
#include "program.h"
#include "matlabprogram.h"
#include <string.h>
#include <exception>
#include "matrix.h"
#include "mex.h"

extern void _main();

inline int round(double x) { 
    return static_cast<int>(x + (x > 0.0 ? +0.5 : -0.5)); 
}


// Constants.
// -----------------------------------------------------------------

// Function definitions. 
// -----------------------------------------------------------------
void mexFunction (int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray *prhs[]) 
  try {

    bool useFuncPointer;
    int i, minNumInputArgs = 4;
    const mxArray *auxData=0, *ptr;
    bool debug = false;

    MatlabString *iterFunc=0;
    MatlabString *objFunc=0, *gradFunc=0;
    const mxArray *ptrFunc=0, *ptrIterFunc=0;

    int    maxIter = defaultmaxiter;
    int maxFnCall = def_maxFuncEval;
    int    m       = defaultm;
    double factr   = defaultfactr;
    double pgtol   = defaultpgtol;

    // Check to see if we have the correct number of input and output
    // arguments.
    if (nrhs < minNumInputArgs)
      throw MatlabException("Incorrect number of input arguments");

    // Get the starting point for the variables. This is specified in
    // the first input argument. The variables must be either a single
    // matrix or a cell array of matrices.
    int k = 0;  // The index of the current input argument.
    ArrayOfMatrices x0(prhs[k++]);

    // Create the output, which stores the solution obtained from
    // running IPOPT. There should be as many output arguments as cell
    // entries in X.
    if (nlhs < x0.length())
      throw MatlabException("Incorrect number of output arguments");
    ArrayOfMatrices x(plhs,x0);

    // Load the lower and upper bounds on the variables as
    // ArrayOfMatrices objects. They should have the same structure as
    // the ArrayOfMatrices object "x".
    ArrayOfMatrices lb(prhs[k++]);
    ArrayOfMatrices ub(prhs[k++]);

    // Check to make sure the bounds make sense.
    if (lb != x || ub != x)
      throw MatlabException("Input arguments LB and UB must have the same \
structure as X");

    // Get the Matlab callback functions.
    if(mxIsChar(prhs[k]))    {
        minNumInputArgs = 5;
        if (nrhs < minNumInputArgs)
            throw MatlabException("Insufficient number of input arguments");

        objFunc = new MatlabString(prhs[k++]);
        gradFunc = new MatlabString(prhs[k++]);
        useFuncPointer = false;
    }
    else    {
        useFuncPointer = true;
        ptrFunc = prhs[k++];
    }

    bool iter = false;

    // Get the auxiliary data.       
    if (nrhs > minNumInputArgs)   {      
        if (!mxIsEmpty(prhs[k++]))
            auxData = prhs[k-1];
    }

    // Get the intermediate callback function.    
    if (nrhs > minNumInputArgs + 1) {
        ptr = prhs[k++];
        if (!mxIsEmpty(ptr))
        {            
            iter = true;
            if (useFuncPointer) 
                ptrIterFunc = ptr;
            else
                iterFunc = new MatlabString(ptr);
        }
    }

//     mexPrintf("useFuncPointer = %d, auxData = %d, iter = %d\n", 
//             useFuncPointer, auxData?1:0, iter);

    // Set the options for the L-BFGS algorithm to their defaults.
    // Process the remaining input arguments, which set options for
    // the IPOPT algorithm.
    if (k < nrhs) 
    {
        for(i = 0; i < mxGetNumberOfFields(prhs[k]); i++) 
        {
            const char * fname = mxGetFieldNameByNumber(prhs[k], i); // field name
            mxArray *tmp = mxGetFieldByNumber(prhs[k], 0, i);        // field value

            if (strcmp(fname, "maxIter") == 0)   {
                maxIter = round(*((double*)mxGetData(tmp)));
                if (maxIter < 0)
                    mexErrMsgTxt("maxIter must be > 0.0.\n");
                if(debug) mexPrintf("Got maxIter = %d\n", maxIter);
            }
            else if (strcmp(fname, "maxFnCall") == 0)   {
                maxFnCall = round(*((double*)mxGetData(tmp)));
                if (maxFnCall < 0)
                    mexErrMsgTxt("maxFnCall must be > 0.0.\n");
                if(debug) mexPrintf("Got maxFuncEval = %d\n", maxFnCall);
            }
            else if (strcmp(fname, "m") == 0)   {
                m = round(*((double*)mxGetData(tmp)));
                if (m < 1)
                    mexErrMsgTxt("m must be > 0.0.\n");
                if(debug) mexPrintf("Got m = %d\n", m);
            }
            else if (strcmp(fname, "relCha") == 0)   {
                factr = *((double*)mxGetData(tmp));
                if (factr < 1e-9)
                    mexErrMsgTxt("relCha must be > 0.0.\n");
                if(debug) mexPrintf("Got relCha = %g\n", factr);
            }
            else if (strcmp(fname, "tolPG") == 0)   {
                pgtol = *((double*)mxGetData(tmp));
                if (pgtol < 1e-9)
                    mexErrMsgTxt("tolPG must be > 0.0.\n");
                if(debug) mexPrintf("Got tolPG = %g\n", pgtol);
            }
            else {
                if(mxIsChar(tmp))    {
                    int buflen = (mxGetM(tmp) * mxGetN(tmp)) + 1;
                    char argBuf[2048];
                    mxGetString(tmp, argBuf, buflen);
                    if(debug) mexPrintf("Unrecognized param: %s = %s\n", fname, argBuf);
                }
                else     {
                    if(debug) mexPrintf("Unrecognized param: %s = %f\n", fname, 
                        *((double*)mxGetData(tmp)));
                }
            }
        }
    }

    // Create a new instance of the optimization problem.
    x = x0;

    MatlabProgram *program;

    if (useFuncPointer)
        program = new MatlabProgram(x,lb,ub,(mxArray*) ptrFunc,(mxArray*) ptrIterFunc,
                        (mxArray*) auxData, m,maxIter,maxFnCall,factr,pgtol);    
    else
        program = new MatlabProgram(x,lb,ub,objFunc,gradFunc,iterFunc,
			            (mxArray*) auxData,m,maxIter,maxFnCall,factr,pgtol);    

    // Run the L-BFGS-B solver.
    
//    mexPrintf("Starting to L-BFGS-B solver\n");
    
    SolverExitStatus exitStatus = program->runSolver();

//     if (exitStatus == abnormalTermination) {
//       if (iterFunc) delete iterFunc;
//       throw MatlabException("Solver unable to satisfy convergence \
// criteria due to abnormal termination");
//     }
//     else if (exitStatus == errorOnInput) {
//       if (iterFunc) delete iterFunc;
//       throw MatlabException("Invalid inputs to L-BFGS routine");
//     }

    // Free the dynamically allocated memory.

    i = x0.length();
    if(nlhs > i)
    {
        plhs[i] = mxCreateDoubleScalar(0.0);
        *mxGetPr(plhs[i]) = program->getOptf();
    }
    if(nlhs > i+1)
    {
        plhs[i+1] = mxCreateDoubleScalar(0.0);
        *mxGetPr(plhs[i+1]) = program->getNumIter();
    }
    if(nlhs > i+2)
    {
        plhs[i+2] = mxCreateDoubleScalar(0.0);
        *mxGetPr(plhs[i+2]) = program->getNumFuncEval();
    }
    if(nlhs > i+3)
    {
        plhs[i+3] = mxCreateDoubleScalar(0.0);
        *mxGetPr(plhs[i+3]) = program->getExitStatus();
    }    

    delete program;
    if (iterFunc)   delete iterFunc;
    if (objFunc)    delete objFunc;
    if (gradFunc)   delete gradFunc;

  } catch (std::exception& error) {
    mexErrMsgTxt(error.what());
  }

