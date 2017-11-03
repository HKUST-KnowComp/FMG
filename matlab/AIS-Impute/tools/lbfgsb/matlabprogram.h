#ifndef INCLUDE_MATLABPROGRAM
#define INCLUDE_MATLABPROGRAM


#include "program.h"
#include "matlabscalar.h"
#include "matlabstring.h"
#include "arrayofmatrices.h"
#include "mex.h"

// Class MatlabProgram.
// -----------------------------------------------------------------
// This is an implementation of the abstract class Program.
class MatlabProgram: public Program {
public:

  // On input, "variables" should contain the initial point for the
  // optimization routine. If no iterative callback function is
  // specified, "iterFunc" may be set to 0. Also, if no auxiliary data
  // is needed, it may also be set to 0.
  // Use function name string
  MatlabProgram (ArrayOfMatrices& variables, 
		 const ArrayOfMatrices& lowerbounds, 
		 const ArrayOfMatrices& upperbounds, 
		 const MatlabString* objFunc, const MatlabString* gradFunc, 
		 const MatlabString* iterFunc, mxArray* auxData, 
		 int m = defaultm, int maxiter = defaultmaxiter,
         int maxFuncIter = def_maxFuncEval,
		 double factr = defaultfactr, double pgtol = defaultpgtol);

  // Use function pointer
  MatlabProgram (ArrayOfMatrices& variables, 
          const ArrayOfMatrices& lowerbounds, 
          const ArrayOfMatrices& upperbounds, 
          mxArray* ptrFunc, 
          mxArray* ptrIterFunc, 
          mxArray* auxData, int m, int maxiter, 
          int maxFuncIter,
          double factr, double pgtol);

  // The destructor.
  virtual ~MatlabProgram();

  void Initialize (ArrayOfMatrices& variables, 
          const ArrayOfMatrices& lowerbounds, 
          const ArrayOfMatrices& upperbounds, 
          mxArray* auxData, int m, int maxiter, 
          double factr, double pgtol);

  // These provide definitions for the pure virtual functions of the
  // abstract parent class.
  virtual double computeObjAndGradient (int n, double* x, double* g);
  virtual void   iterCallback     (int t, double* x, double f);

  // An implementation of this method should return the value of the
  // objective at the current value of the variables "x", where "n" is
  // the number of variables.
  double computeObjective (int n, double* x);

  // An implementation of this method should fill in the values of the
  // gradient "g" and the current point "x".
  void   computeGradient  (int n, double* x, double* g);  


  // Run the solver. Upon completion, the solution is stored in
  // "variables".
  SolverExitStatus runSolver();

protected:
  ArrayOfMatrices&    variables;  // Storage for the initial value and 
                                  // solution.
  const MatlabString* objFunc;    // The objective callback function.
  const MatlabString* gradFunc;   // The gradient callback function.
  const MatlabString* iterFunc;   // The iterative callback function.

  mxArray *ptrFunc, *ptrIterFunc; // Function pointers

  bool useFuncPointer;
  mxArray **inputIterRhs, **inputGradRhs;


  ArrayOfMatrices*    varMatlab;  // Inputs to the Matlab callback
  mxArray**           varInputs;  // functions representing the
				  // current values of the variables.
  MatlabScalar*       fMatlab;    // Input to the Matlab callback
				  // functions representing the
				  // current value of the objective.
  MatlabScalar*       tMatlab;    // Input to the Matlab callback
				  // functions representing the
				  // current iteration.

  int       numInputsObjFunc;  // The number of inputs passed to the
			       // objective callback function.
  int       numInputsGradFunc; // The number of inputs passed to the
			       // gradient callback function.
  int       numInputsIterFunc; // The number of inputs passed to the
			       // iterative callback function.
  mxArray** inputsObjFunc;     // The inputs passed to the objective 
                               // callback function.
  mxArray** inputsGradFunc;    // The inputs passed to the gradient
			       // callback function.
  mxArray** inputsIterFunc;    // The inputs passed to the iterative
			       // callback function.
};

#endif
