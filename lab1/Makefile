IDIR=include
#Choose compiler
CXX ?= g++
CXXFLAGS=-I$(IDIR) -std=c++11 -fno-omit-frame-pointer -g -O2 -march=native -flto -fopt-info-loop-optimized=loop.opt

ODIR=src
LDIR =../lib

LIBS=-lm -fopenmp 

_DEPS = deep_core.h vector_ops.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = deep_core.o vector_ops.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< ${CXXFLAGS}

nnetwork_mpi.o:
	$(CXX) -c -o $@ nnetwork.cxx ${CXXFLAGS} -DUSE_MPI

nnetwork.o:
	$(CXX) -c -o $@ nnetwork.cxx ${CXXFLAGS} 

nnetwork_mpi: $(OBJ) nnetwork_mpi.o
	$(CXX) -o $@ $^ ${CXXFLAGS} $(LIBS)

nnetwork: $(OBJ) nnetwork.o
	$(CXX) -o $@ $^ ${CXXFLAGS} $(LIBS)

run_serial:
	./nnetwork

run_perf:
	timeout 180 perf record -g --call-graph fp ./nnetwork

run_llc: nnetwork
	perf stat -e LLC-loads,LLC-load-misses ./nnetwork
	
all: clean nnetwork_mpi nnetwork
.PHONY: clean
default: clean nnetwork 

.DEFAULT_GOAL := default

clean:
	rm -f $(ODIR)/*.o *.o nnetwork_mpi nnetwork
