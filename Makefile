include Makefile.config

# Where to place outputs
BINDIR = bin
LIBDIR = lib

# Where to find inputs
SRCDIR = src

# Where to do intermediate stuff
BUILDDIR = build
DEPSDIR = $(BUILDDIR)

# Targets to build
TARGETS = $(LIBDIR)/libcprof.so

# Source and object files
CPP_SRCS := $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/**/*.cpp) 
CPP_OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(CPP_SRCS))
CPP_DEPS=$(patsubst $(BUILDDIR)/%.o,$(DEPSDIR)/%.d,$(CPP_OBJECTS))
DEPS = $(CPP_DEPS)


INC += -Iinclude/cprof
LIB += -L$(LIBDIR)

# Use BOOST_ROOT if set
ifdef BOOST_ROOT
  BOOST_INC=$(BOOST_ROOT)/include
  BOOST_LIB=$(BOOST_ROOT)/lib
  INC += -isystem$(BOOST_INC)
  LIB += -L$(BOOST_LIB)
endif

CXX = g++
ifdef CUDA_ROOT
	NVCC = $(CUDA_ROOT)/bin/nvcc
else
	NVCC = nvcc
endif

LD = ld
CXXFLAGS += -std=c++11 -g -fno-omit-frame-pointer -Wall -Wextra -Wshadow -Wpedantic -fPIC
NVCCFLAGS += -std=c++11 -g -arch=sm_35 -Xcompiler -Wall,-Wextra,-fPIC,-fno-omit-frame-pointer
INC += -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB += -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti \
      -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt \
      -ldl -lnuma \
	  -L/usr/include/boost

all: $(TARGETS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/* $(LIBDIR)/*

.PHONY: distclean
disclean: clean
	rm -rf $(BINDIR)
	rm -rf $(LIBDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(DEPSDIR)

$(LIBDIR)/libcprof.so: $(CPP_OBJECTS)
	mkdir -p $(LIBDIR)
	$(CXX) -shared $^ -o $@ $(LIB)

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	cppcheck $< 
	mkdir -p `dirname $@`
	$(CXX) -MMD -MP $(CXXFLAGS) $(INC) $< -c -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p `dirname $@`
	$(NVCC) -std=c++11 -arch=sm_35 -dc  -Xcompiler -fPIC $^ -o test.o
	$(NVCC) -std=c++11 -arch=sm_35 -Xcompiler -fPIC -dlink test.o -lcudadevrt -lcudart -o $@	

.PHONY: docs
docs:
	mkdir -p docs
	doxygen doxygen.config
	make -C docs/latex

-include $(DEPS)



#prof.so: $(OBJECTS)
#	$(CXX) -shared $(LIB) $^ test.o -o $@
#%.o : %.cu
#	$(NVCC) $(NVCCFLAGS) -dc $^ -lcudadevrt -lcudart -o $@	

