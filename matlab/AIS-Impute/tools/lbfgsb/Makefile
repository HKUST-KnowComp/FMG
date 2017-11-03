# Linux settings.
ifneq (,$(findstring Linux,$(OSTYPE)))
MEX         = /opt/matlab.R2008a/bin/mex
MATLAB_HOME = /opt/matlab.R2008a
CXX         = g++
F77         = gfortran
CFLAGS      = -O3 -fPIC -pthread 
FFLAGS      = -O3 -fPIC -fexceptions 
LDFORTRAN   = 
endif

# Mac OS X settings.
#ifneq (,$(findstring darwin,$(OSTYPE)))
MEX         = mex
MATLAB_HOME = /Applications/MATLAB_R2011a.app
CXX         = g++
F77         = gfortran 
CFLAGS      = -O3 -fPIC -fno-common -fexceptions -no-cpp-precomp 
FFLAGS      = -O3 -x f77-cpp-input -fPIC -fno-common 
LDFORTRAN   = -L/usr/local/gfortran/lib
#endif

TARGET = lbfgsb

OBJS   = solver.o matlabexception.o matlabscalar.o matlabstring.o   \
         matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o \
         lbfgsb.o
         
CFLAGS += -Wall -ansi -DMATLAB_MEXFILE

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(MATLAB_HOME)/extern/include -o $@ -c $^

%.o: %.f
	$(F77) $(FFLAGS) -o $@ -c $^

$(TARGET): $(OBJS)
	$(MEX) -cxx CXX=$(CXX) CC=$(CXX) FC=$(FCC) LD=$(CXX) $(LDFORTRAN) -lgfortran -lm \
        -O -output $@ $^

clean:
	rm -f *.o $(TARGET).mex*

