/* -------------------------------------------------------------------------- */
/* partXY_mex mexFunction */
/* -------------------------------------------------------------------------- */

#include "mex.h"
#include "blas.h"

/* compute a part of X*Y */
void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin  [ ]
)
{
    if (nargin != 5 || nargout > 1)
        mexErrMsgTxt ("Usage: v = partXY (Xt, Y, I, J, L)") ;

    /* ---------------------------------------------------------------- */
    /* inputs */
    /* ---------------------------------------------------------------- */
    
    double *Xt = mxGetPr( pargin [0] );
    double *Y  = mxGetPr( pargin [1] );
    double *I  = mxGetPr( pargin [2] );
    double *J  = mxGetPr( pargin [3] );
    double  LL = mxGetScalar( pargin [4] ); 
    
    ptrdiff_t L = (ptrdiff_t) LL;
    ptrdiff_t m = mxGetN( pargin [0] );
    ptrdiff_t n = mxGetN( pargin [1] );
    ptrdiff_t r = mxGetM( pargin [0] ); 
    
    if ( r != mxGetM( pargin [1] ))
        mexErrMsgTxt ("rows of Xt must be equal to rows of Y") ;
    if ( r > m || r > n )
        mexErrMsgTxt ("rank must be r <= min(m,n)") ;
    
    /* ---------------------------------------------------------------- */
    /* output */
    /* ---------------------------------------------------------------- */

    pargout [0] = mxCreateDoubleMatrix(1, L, mxREAL);
    double *v = mxGetPr( pargout [0] );
    ptrdiff_t inc = 1;
    
    /* C array indices start from 0 */
    for (ptrdiff_t p = 0; p < L; p++) 
    {
        ptrdiff_t ir = ( I[p] - 1 ) * r;
        ptrdiff_t jr = ( J[p] - 1 ) * r;
        v[p] = ddot(&r, Xt+ir, &inc, Y+jr, &inc);
    }
    
    return;
}

