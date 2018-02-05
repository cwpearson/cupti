include ../Makefile.config
include Makefile.config

# Where to place outputs
LIBDIR = lib

# Where to find inputs
SRCDIR = src

# Where to do intermediate stuff
BUILDDIR = build
DEPSDIR = deps

# How to build tests
TEST_DIR = tests
include $(TEST_DIR)/Makefile

# Targets to build
TARGETS = $(LIBDIR)/libcprof.so

# Source and object files
CPP_SRCS := $(shell find src -name "*.cpp")
CPP_OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(CPP_SRCS))
CPP_DEPS=$(patsubst $(BUILDDIR)/%.o,$(DEPSDIR)/%.d,$(CPP_OBJECTS))
DEPS = $(CPP_DEPS)

INC += -Iinclude
LIB += -L$(LIBDIR)

ifdef BOOST_ROOT
  BOOST_INC=$(BOOST_ROOT)/include
  BOOST_LIB=$(BOOST_ROOT)/lib
  INC += -isystem$(BOOST_INC) 
  LIB += -L$(BOOST_LIB) -Wl,-rpath=$(BOOST_LIB)
endif
LIB += -lboost_system


ifdef NUMA_ROOT
  NUMA_INC=$(NUMA_ROOT)/include
  NUMA_LIB=$(NUMA_ROOT)/lib
  INC += -isystem$(NUMA_INC) 
  LIB += -L$(NUMA_LIB) -Wl,-rpath=$(BOOST_LIB)
endif
LIB += -lnuma

# Set CUDA-related variables
ifdef CUDA_ROOT
  CUDA_CUPTI_LIB = $(CUDA_ROOT)/extras/CUPTI/lib64
else
  $(error set CUDA_ROOT in Makefile.config)
endif
NVCC = $(CUDA_ROOT)/bin/nvcc
INC += -isystem$(CUDA_ROOT)/include -isystem$(CUDA_ROOT)/extras/CUPTI/include
LIB += -L$(CUDA_CUPTI_LIB) -lcupti -Wl,-rpath=$(CUDA_CUPTI_LIB) \
       -L$(CUDA_ROOT)/lib64 -lcuda -lcudart -lcudadevrt

CXXFLAGS += -std=c++11 -Wall -Wextra -Wshadow -Wpedantic -fPIC
NVCCFLAGS += -std=c++11 -arch=sm_35 -Xcompiler -Wall,-Wextra,-fPIC

ifeq ($(BUILD_TYPE),Release)
  CXXFLAGS +=  -Ofast
  NVCCFLAGS += -Xcompiler -Ofast
else ifeq ($(BUILD_TYPE),Debug)
  CXXFLAGS += -g -fno-omit-frame-pointer
  NVCCFLAGS += -G -g -arch=sm_35 -Xcompiler -fno-omit-frame-pointer
else
  $(error BUILD_TYPE must be Release or Debug)
endif


# Since this is not the first target, use .DEFAULT_GOAL
.DEFAULT_GOAL := all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/* $(LIBDIR)/* $(DEPSDIR)/* $(CLEAN_TARGETS)

.PHONY: distclean
distclean: clean
	rm -rf $(LIBDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(DEPSDIR)

.PHONY: cppcheck
cppcheck:
	cppcheck --enable=all src $(INC)

$(LIBDIR)/libcprof.so: $(CPP_OBJECTS)
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ $(LIB) -Wl,--no-undefined

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	mkdir -p `dirname $(patsubst $(BUILDDIR)/%.o,$(DEPSDIR)/%.d,$@)`
	mkdir -p `dirname $@`
	$(CXX) -MMD -MP -MF $(patsubst $(BUILDDIR)/%.o,$(DEPSDIR)/%.d,$@) $(CXXFLAGS) $(INC) $< -c -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	mkdir -p `dirname $@`
	$(NVCC) -std=c++11 -arch=sm_35 -dc  -Xcompiler -fPIC $^ -o test.o
	$(NVCC) -std=c++11 -arch=sm_35 -Xcompiler -fPIC -dlink test.o -lcudadevrt -lcudart -o $@	

.PHONY: docs docker_docs
docs:
	doxygen doxygen.config
	make -C docs/latex
docker_docs:
	@docker pull cwpearson/doxygen
	@docker run -it --rm -v `pwd`:/data cwpearson/doxygen  doxygen doxygen.config
	@docker run -it --rm -v `readlink -f docs/latex`:/data cwpearson/doxygen make

docker_build: Dockerfile
	@docker build . -t cwpearson/cprof

-include $(DEPS)

#prof.so: $(OBJECTS)
#	$(CXX) -shared $(LIB) $^ test.o -o $@
#%.o : %.cu
#	$(NVCC) $(NVCCFLAGS) -dc $^ -lcudadevrt -lcudart -o $@	


