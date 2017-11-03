function msg_str = msg2str(msg, solver_name)

  if strcmp(solver_name, 'pbm') == 1
    switch msg
      case 0
          msg_str = 'The problem has been solved.';
      case 1
          msg_str = 'Number of calls of function = maxFnCall.';
      case 2
          msg_str = 'Number of iterations = maxIter.';
      case 3
          msg_str = 'Invalid input parameters.';
      case 4
          msg_str = 'Not enough working space.';
      case 5
          msg_str = 'Failure in quadratic program';
      case 6
          msg_str = 'The starting point is not feasible.';
      case 7
          msg_str = 'Failure in attaining the demanded accuracy.';
      case 8
          msg_str = 'Failure in function or subgradient calculations.';
      otherwise
          msg_str = ['Unknown exit code: ' num2str(msg) 'in pbm.'];
    end
  end


  if strcmp(solver_name, 'lbfgsb') == 1
    switch msg
      case 1
          msg_str = 'The problem has been solved.';
      case 2
          msg_str = 'The relative change of obj value < relative change.';
      case 3
          msg_str = 'The norm of projected gradient < tolPG.';
      case 4
          msg_str = 'Number of function calls > maxFnCall.';
      case 5
          msg_str = 'Number of iterations > maxIter';
      case -1
          msg_str = 'The routine has detected an error in the input parameters.';
      case -2
          msg_str = 'terminated abnormally and was unable to satisfy the convergence criteria.';
      otherwise
          msg_str = ['Unknown exit code: ' num2str(msg) ' in lbfgsb.'];
    end
  end

  if strcmp(solver_name, 'liblbfgs') == 1
    switch msg
      case {0,1}
          msg_str = 'Success!';
      case 2
          msg_str = 'Initial variables already minimze the objective function.';
      case -1024
          msg_str = 'Unknown error.';
      case -1023
          msg_str = 'Logic error.';
      case -1022
          msg_str = 'Out of memory';
      case -1021
          msg_str = 'The minimization process has been canceled.';
      case -1020
          msg_str = 'Invalid number of variables specified.';
      case -1019
          msg_str = 'Invalid number of variables (for SSE) specified.';
      case -1018
          msg_str = 'The array x must be aligned to 16 (for SSE).';
      case -1017
          msg_str = 'Invalid parameter epsilon specified.';
      case -1016
          msg_str = 'Invalid parameter past specified.';
      case -1015
          msg_str = 'Invalid parameter delta specified.';
      case -1014
          msg_str = 'Invalid parameter linesearch specified.';
      case -1013
          msg_str = 'Invalid parameter min_step specified.';
      case -1012
          msg_str = 'Invalid parameter max_step specified.';
      case -1011
          msg_str = 'Invalid parameter ftol specified.';
      case -1010
          msg_str = 'Invalid parameter wolfe specified.';
      case -1009
          msg_str = 'Invalid parameter gtol specified.';
      case -1008
          msg_str = 'Invalid parameter xtol specified.';
      case -1007
          msg_str = 'Invalid parameter max_linesearch specified.';
      case -1006
          msg_str = 'Invalid parameter orthantwise_c specified.';
      case -1005
          msg_str = 'Invalid parameter orthantwise_start specified.';
      case -1004
          msg_str = 'Invalid parameter orthantwise_end specified.';
      case -1003
          msg_str = 'The line-search step went out of the interval of uncertainty.';
      case -1002
          msg_str = 'A logic error occurred; alternatively, the interval of uncertainty became too small.';
      case -1001
          msg_str = 'A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions.';
      case -1000
          msg_str = 'The line-search step became smaller than min_step.';
      case -999
          msg_str = 'The line-search step became larger than max_step.';
      case -998
          msg_str = 'The line-search routine reaches the maximum number of evaluations.';
      case -997
          msg_str = 'The maximum number of iterations was reached.';
      case -996
          msg_str = 'Relative width of the interval of uncertainty is at most xtol.';
      case -995
          msg_str = 'A logic error (negative line-search step) occurred.';
      case -994
          msg_str = 'The current search direction increases the objective function value.';
      otherwise
          msg_str = ['Unknown exit code: ' num2str(msg) ' in liblbfgs.'];
    end
  end
end
