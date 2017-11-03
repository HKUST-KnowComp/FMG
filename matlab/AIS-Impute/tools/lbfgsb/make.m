
if strcmp(computer(), 'PCWIN') == 1 % on windows machine

    cmd = 'mex -f C:\Users\xzhang\AppData\Roaming\MathWorks\MATLAB\R2011a\mexopts_ivf.bat -c ';
    
    eval([cmd 'solver.f']);
    
    cmd_root = 'mex -f C:\Users\xzhang\AppData\Roaming\MathWorks\MATLAB\R2011a\mexopts_vc2010.bat ';    % or mexopts_vc2010.bat
    cmd = [cmd_root '-c '];
    
    eval([cmd 'matlabexception.cpp']);
    eval([cmd 'matlabscalar.cpp']);
    eval([cmd 'matlabstring.cpp']);
    eval([cmd 'matlabmatrix.cpp']);
    eval([cmd 'arrayofmatrices.cpp']);
    eval([cmd 'program.cpp']);
    eval([cmd 'matlabprogram.cpp']);
    eval([cmd 'lbfgsb.cpp']);

    eval([cmd_root '-output lbfgsb solver.obj matlabexception.obj matlabscalar.obj matlabstring.obj matlabmatrix.obj arrayofmatrices.obj program.obj matlabprogram.obj lbfgsb.obj']);
    
else
    
    % The simplest way is just: system('make all');
    % But let us do it in a different way
    cmd = 'system (''gfortran -fPIC -c ';
    
    eval([cmd 'solver.f'')']);
    
    cmd = 'mex -c ';
    
    eval([cmd 'matlabexception.cpp']);
    eval([cmd 'matlabscalar.cpp']);
    eval([cmd 'matlabstring.cpp']);
    eval([cmd 'matlabmatrix.cpp']);
    eval([cmd 'arrayofmatrices.cpp']);
    eval([cmd 'program.cpp']);
    eval([cmd 'matlabprogram.cpp']);
    eval([cmd 'lbfgsb.cpp']);
       
    mex -cxx -lgfortran -output lbfgsb matlabexception.o matlabscalar.o matlabstring.o matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o lbfgsb.o solver.o
end

copyfile('lbfgsb.mexmaci64', 'lbfgsb2.mexmaci64', 'f');
