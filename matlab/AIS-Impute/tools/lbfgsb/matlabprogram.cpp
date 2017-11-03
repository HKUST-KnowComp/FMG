#include "matlabprogram.h"
#include "array.h"

// Function definitions.
// -----------------------------------------------------------------
// Copy the matrix elements to the destination array. It is assumed
// that sufficient memory has been allocated for the destination
// array. The copying preserves the column-major ordering of the
// matrix elements.
void copyElemsFromMatrix (const Matrix& source, double* dest) {
  Matrix destmatrix(dest,source.height(),source.width());
  destmatrix = source;
}

// Copy the elements from the source array into the matrix.
void copyElemsToMatrix (const double* source, Matrix& dest) {
  dest.inject(source);
}

// Copy the elements contained in all the matrices to the destination
// array. It is assumed that sufficient memory has been allocated for
// the destination array. The copying preserves the column-major
// ordering of the matrix elements.
void copyElemsFromArrayOfMatrices (const ArrayOfMatrices& source, 
				   double* dest) {

  // Repeat for each matrix.
  for (int i = 0; i < source.length(); i++) {
    copyElemsFromMatrix(*source[i],dest);
    dest += source[i]->length();
  }
}

// Copy the elements from the source array into the array of matrices.
void copyElemsToArrayOfMatrices (const double* source, 
				 ArrayOfMatrices& dest) {

  // Repeat for each matrix.
  for (int i = 0; i < dest.length(); i++) {
    copyElemsToMatrix(source,*dest[i]);
    source += dest[i]->length();
  }
}

int getBoundType (double& lb, double& ub) {
  bool haslb = mxIsFinite(lb);
  bool hasub = mxIsFinite(ub);
  int  btype = haslb + 3*hasub - 2*(haslb && hasub);

  lb = haslb ? lb : 0;
  ub = hasub ? ub : 0;

  return btype;
}


// Function definitions for class MatlabProgram.
// -----------------------------------------------------------------
void MatlabProgram::Initialize (ArrayOfMatrices& variables, 
			      const ArrayOfMatrices& lowerbounds, 
			      const ArrayOfMatrices& upperbounds, 
			      mxArray* auxData, int m, int maxiter,
			      double factr, double pgtol) 
{
  int i;
    
  x     = new double[n];
  lb    = new double[n];
  ub    = new double[n];
  btype = new int[n];  

  // Copy some pieces of information from the input arguments to the
  // respective program structures.
  copyElemsFromArrayOfMatrices(variables,x);  
  copyElemsFromArrayOfMatrices(lowerbounds,lb);
  copyElemsFromArrayOfMatrices(upperbounds,ub);

  // Set the bound types.
  for (i = 0; i < n; i++)
    btype[i] = getBoundType(lb[i],ub[i]);

  // Create a Matlab array for the variables.
  int nv    = variables.length();
  varInputs = new mxArray*[nv];
  varMatlab = new ArrayOfMatrices(varInputs,variables);

  // Set up the inputs to the objective callback function.
  numInputsObjFunc = nv + (bool) auxData;
  inputsObjFunc    = new mxArray*[numInputsObjFunc];
  copymemory(varInputs,inputsObjFunc,nv);
  if (auxData)
    inputsObjFunc[nv] = auxData;

  // Set up the inputs to the gradient callback function.
  numInputsGradFunc = numInputsObjFunc;
  inputsGradFunc    = inputsObjFunc;

  if(useFuncPointer)
  {
      inputGradRhs = new mxArray*[numInputsGradFunc+1];
      inputGradRhs[0] = ptrFunc;
      for (i = 0; i<numInputsGradFunc; i ++)
          inputGradRhs[i+1] = inputsObjFunc[i];
  }

  // If necessary, set up the inputs to the iterative callback
  // function. The first two inputs are the iteration and the function
  // objective. The next of inputs are the values of the variables
  // then, optionally, the auxiliary data.
  if (iterFunc || ptrIterFunc) {
    numInputsIterFunc = 2 + nv + (bool) auxData;
    inputsIterFunc    = new mxArray*[numInputsIterFunc];
    mxArray** ptr     = inputsIterFunc;
    
    tMatlab = new MatlabScalar(*ptr++,0);
    fMatlab = new MatlabScalar(*ptr++,0);
    copymemory<mxArray*>(varInputs,ptr,nv);
    ptr += nv;
    if (auxData)
      *ptr = auxData;

    if(useFuncPointer)
    {
        inputIterRhs = new mxArray*[numInputsIterFunc+1];
        inputIterRhs[0] = ptrIterFunc;
        for (i = 0; i<numInputsIterFunc; i ++)
            inputIterRhs[i+1] = inputsIterFunc[i];
    }
  }
}


// Function definitions for class MatlabProgram.
// -----------------------------------------------------------------
MatlabProgram::MatlabProgram (ArrayOfMatrices& variables, 
			      const ArrayOfMatrices& lowerbounds, 
			      const ArrayOfMatrices& upperbounds, 
			      const MatlabString* objFunc, 
			      const MatlabString* gradFunc, 
			      const MatlabString* iterFunc, 
			      mxArray* auxData, int m, int maxiter, int maxFuncIter,
			      double factr, double pgtol) 
  : Program(variables.numelems(),0,0,0,0,m,maxiter,maxFuncIter,factr,pgtol),
    variables(variables) {

  this->objFunc  = objFunc;
  this->gradFunc = gradFunc;
  this->iterFunc = iterFunc;
  this->inputIterRhs = 0;
  this->inputGradRhs = 0;

  this->ptrFunc  = 0;
  this->ptrIterFunc = 0;

  useFuncPointer = false;

  if(!gradFunc || !objFunc)
      mexErrMsgTxt("Routines of function and gradient evaluation mis-specified.\n");

  Initialize (variables, lowerbounds, upperbounds, auxData, m, maxiter, factr, pgtol);
}


// Function definitions for class MatlabProgram.
// -----------------------------------------------------------------
MatlabProgram::MatlabProgram (ArrayOfMatrices& variables, 
                   const ArrayOfMatrices& lowerbounds, 
                   const ArrayOfMatrices& upperbounds, 
                   mxArray* ptrFunc, 
                   mxArray* ptrIterFunc, 
                   mxArray* auxData, int m, int maxiter, int maxFuncIter,
                   double factr, double pgtol) 
   : Program(variables.numelems(),0,0,0,0,m,maxiter,maxFuncIter,factr,pgtol),
      variables(variables) {

    this->objFunc  = 0;
    this->gradFunc = 0;
    this->iterFunc = 0;
    this->inputIterRhs = 0;
    this->inputGradRhs = 0;

    this->ptrFunc  = ptrFunc;
    this->ptrIterFunc = ptrIterFunc;

    if(!ptrFunc)
        mexErrMsgTxt("Routine of function and gradient evaluation mis-specified.\n");

    useFuncPointer = true;

    Initialize (variables, lowerbounds, upperbounds, auxData, m, maxiter, factr, pgtol);
}


MatlabProgram::~MatlabProgram() {
  delete[] x;
  delete[] lb;
  delete[] ub;
  delete[] btype;

  // Deallocate the Matlab arrays.
  int nv = variables.length();
  for (int i = 0; i < nv; i++)
    mxDestroyArray(varInputs[i]);
  delete   varMatlab;
  delete[] varInputs;
  if (inputGradRhs)  delete [] inputGradRhs;
  if (inputIterRhs)  delete [] inputIterRhs;

  // If necessary, deallocate the additional Matlab arrays that act as
  // inputs to the iterative callback function.
  if (iterFunc) {
    delete   fMatlab;
    delete   tMatlab;
    // delete[] inputsIterFunc;
  }
}

double MatlabProgram::computeObjective (int n, double* x) {
  int      nlhs = 1;    // The number of outputs from Matlab.
  mxArray** plhs;  // The outputs from the Matlab routine.

  plhs = new mxArray * [nlhs];
  // Copy the current value of the optimization variables.
  copyElemsToArrayOfMatrices(x,*varMatlab); 

  // Call the designated Matlab routine for evaluating the objective
  // function. It takes as input the values of the variables, and
  // returns a single output, the value of the objective function.

  if (mexCallMATLAB(nlhs,plhs,numInputsObjFunc,inputsObjFunc,*objFunc))
    throw MatlabException("Evaluation of objective function failed in \
call to MATLAB routine");

  // Get the result passed back from the Matlab routine.
  MatlabScalar matlabOutput(plhs[0]);
  double       objective = matlabOutput;
  
  // Free the dynamically allocated memory. 
  mxDestroyArray(plhs[0]);

  delete [] plhs;
  return objective;
}

void MatlabProgram::computeGradient (int n, double* x, double* g) {
  int      nlhs = variables.length(); // The number of outputs from Matlab.
  mxArray** plhs;                // The outputs from the Matlab routine.

  plhs = new mxArray * [nlhs];

  // Copy the current value of the optimization variables.
  copyElemsToArrayOfMatrices(x,*varMatlab); 

  // Call the designated Matlab routine for computing the gradient
  // of the objective function. It takes as input the values of the
  // variables, and returns as many outputs corresponding to the
  // partial derivatives of the objective function with respect to
  // the variables at their curent values.
  if (mexCallMATLAB(nlhs,plhs,numInputsGradFunc,inputsGradFunc,*gradFunc))
    throw MatlabException("Evaluation of objective gradient failed in \
call to MATLAB routine");

  // Get the result passed back from the Matlab routine.
  ArrayOfMatrices matlabOutput((const mxArray**) plhs,nlhs);
  for (int i = 0; i < variables.length(); i++)
    if (matlabOutput[i]->length() != variables[i]->length())
      throw MatlabException("Invalid gradient passed back from MATLAB \
routine");	
  copyElemsFromArrayOfMatrices(matlabOutput,g);
  
  // Free the dynamically allocated Matlab outputs.
  for (int i = 0; i < variables.length(); i++)
    mxDestroyArray(plhs[i]);
  delete [] plhs;
}


double MatlabProgram::computeObjAndGradient (int n, double* x, double* g)
{
    double f;

    if (!useFuncPointer)
    {
        f = computeObjective(n,x);
        computeGradient(n,x,g);
    }
    else
    {
        int      nlhs = variables.length() + 1; // The number of outputs from Matlab.
        mxArray** plhs;                // The outputs from the Matlab routine.

        plhs = new mxArray * [nlhs];

        // Copy the current value of the optimization variables.
        copyElemsToArrayOfMatrices(x,*varMatlab); 

        // Call the designated Matlab routine for computing the objective and gradient
        // of the objective function. It takes as input the values of the
        // variables, and returns the objective, and as many outputs corresponding 
        // to the partial derivatives of the objective function with respect to
        // the variables at their current values.
        if (mexCallMATLAB(nlhs,plhs,numInputsGradFunc+1,inputGradRhs,"feval"))
            throw MatlabException("Evaluation of objective gradient failed in \
call to MATLAB routine");

        // Get the result passed back from the Matlab routine.
        f = (double)(mxGetScalar(plhs[0]));
        ArrayOfMatrices matlabOutput((const mxArray**) (plhs+1),nlhs-1);
        for (int i = 0; i < variables.length(); i++)
            if (matlabOutput[i]->length() != variables[i]->length())
                throw MatlabException("Invalid gradient passed back from MATLAB \
routine");	
        copyElemsFromArrayOfMatrices(matlabOutput,g);

        // Free the dynamically allocated Matlab outputs.
        for (int i = 0; i <= variables.length(); i++)
            mxDestroyArray(plhs[i]);
        delete [] plhs;
    }

    return f;
}

void MatlabProgram::iterCallback (int t, double* x, double f) {

    if (!iterFunc && !ptrIterFunc)
        return;

    *tMatlab = t;
    *fMatlab = f;
    copyElemsToArrayOfMatrices(x,*varMatlab); 

    if (iterFunc)    {
        // Copy the current iteration, the current value of the objective
        // function, and the current value of the optimization variables.

        // Call the Matlab iterative callback routine. Since there are no
        // outputs, there is no memory to deallocate.
        if (mexCallMATLAB(0,0,numInputsIterFunc,inputsIterFunc,*iterFunc))
            throw MatlabException("Call to MATLAB iterative callback function \
                                  failed");    
    }
    else if (ptrIterFunc)    {                
        if (mexCallMATLAB(0, 0, numInputsIterFunc+1, inputIterRhs, "feval"))
            throw MatlabException("Call to MATLAB iterative callback function \
                                  failed");    
    }
}

SolverExitStatus MatlabProgram::runSolver() {
  SolverExitStatus status;  // The return value.
  status = Program::runSolver();

  // Copy the solution to the class member "variables".
  copyElemsToArrayOfMatrices(x,variables);

  return status;
}
