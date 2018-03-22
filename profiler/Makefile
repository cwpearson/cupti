include ../Makefile.config
include Makefile.config

# Where to place outputs
LIBDIR = lib

# Where to find inputs
SRCDIR = src

# Where to do intermediate stuff
BUILDDIR = build
DEPSDIR = $(BUILDDIR)

# Targets to build
TARGETS = $(LIBDIR)/libprofiler.so

# Source and object files
CPP_SRCS := $(shell find src -name "*.cpp")
CPP_OBJECTS = $(patsubst $(SRCDIR)/%.cpp, $(BUILDDIR)/%.o, $(CPP_SRCS))
CPP_DEPS=$(patsubst $(BUILDDIR)/%.o,$(DEPSDIR)/%.d,$(CPP_OBJECTS))
DEPS = $(CPP_DEPS)

INC += -Iinclude 
LIB += -L$(LIBDIR)

ifdef BOOST_ROOT
  BOOST_INC=$(BOOST_ROOT)/include
  INC += -isystem$(BOOST_INC) 
endif

ifdef OPENTRACING_ROOT 
  OPENTRACING_LIB=$(OPENTRACING_ROOT)/lib
  OPENTRACING_INC=$(OPENTRACING_ROOT)/include
  INC += -isystem$(OPENTRACING_INC)
  LIB += -L$(OPENTRACING_LIB) -Wl,-rpath=$(OPENTRACING_LIB)
endif

ifdef ZIPKIN_OPENTRACING_ROOT
  ZIPKIN_OPENTRACING_LIB = $(ZIPKIN_OPENTRACING_ROOT)/lib
  INC += -isystem$(ZIPKIN_OPENTRACING_ROOT)/include
  LIB += -L$(ZIPKIN_OPENTRACING_LIB) -Wl,-rpath=$(ZIPKIN_OPENTRACING_LIB)
endif

CPROF_LIB=$(CPROF_ROOT)/lib
INC += -isystem$(CPROF_ROOT)/include -isystem$(CPROF_ROOT)/external/include
LIB += -L$(CPROF_LIB) -lcprof -Wl,-rpath=$(CPROF_LIB)

# Set CUDA-related variables
ifndef CUDA_ROOT
  $(error set CUDA_ROOT in Makefile.config)
endif
NVCC = $(CUDA_ROOT)/bin/nvcc
INC += -isystem$(CUDA_ROOT)/include -isystem$(CUDA_ROOT)/extras/CUPTI/include
LIB += -ldl \
       -L$(CUDA_ROOT)/extras/CUPTI/lib64 -lcupti -Wl,-rpath=$(CUDA_ROOT)/extras/CUPTI/lib64 \
       -L$(CUDA_ROOT)/lib64 -lcuda -lcudart -lcudadevrt -lcudnn -L/usr/lib -lnccl \
	     -lopentracing -lzipkin -lzipkin_opentracing

CXXFLAGS += -std=c++11 -Wall -Wextra -Wshadow -Wpedantic -fPIC -pthread
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

all: $(TARGETS)

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/* $(LIBDIR)/*

.PHONY: distclean
distclean: clean
	rm -rf $(LIBDIR)
	rm -rf $(BUILDDIR)
	rm -rf $(DEPSDIR)

.PHONY: cppcheck
cppcheck:
	cppcheck --enable=all src $(INC)

$(LIBDIR)/libprofiler.so: $(CPP_OBJECTS)
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-undefined $^ -o $@ $(LIB)

$(BUILDDIR)/%.o : $(SRCDIR)/%.cpp
	mkdir -p `dirname $@`
	$(CXX) -MMD -MP $(CXXFLAGS) $(INC) $< -c -o $@

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


