include Makefile.config

TARGETS = prof.so

OBJECTS = \
kernel_time.o \
address_space.o \
allocation_record.o \
allocations.o \
api_record.o \
apis.o \
callbacks.o \
cupti_subscriber.o \
driver_state.o \
extent.o \
memory.o \
numa.o \
preload_cublas.o \
preload_cudart.o \
preload_cudnn.o \
thread.o \
value.o \
values.o

DEPS=$(patsubst %.o,%.d,$(OBJECTS))

ifneq ($(BOOST_ROOT),)
  BOOST_INC=$(BOOST_ROOT)/include
  BOOST_LIB=$(BOOST_ROOT)/lib
  INC += -isystem$(BOOST_INC)
  LIB += -L$(BOOST_LIB)
endif

LD = ld
CXX = g++
CXXFLAGS += -std=c++11 -g -fno-omit-frame-pointer -Wall -Wextra -Wshadow -Wpedantic -fPIC
NVCC = nvcc
NVCCFLAGS += -std=c++11 -g -arch=sm_35 -Xcompiler -Wall,-Wextra,-fPIC,-fno-omit-frame-pointer
INC += -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB += -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti \
      -L/usr/local/cuda/lib64 -lcuda -lcudart -lcudadevrt \
      -ldl -lnuma \
	  -L/usr/include/boost

all: $(TARGETS)

clean:
	rm -f $(OBJECTS) $(DEPS) $(TARGETS)

prof.so: $(OBJECTS)
	$(CXX) -shared $^ -o $@ $(LIB)

%.o : %.cpp
	cppcheck $<
	$(CXX) -MMD -MP $(CXXFLAGS) $(INC) $< -c -o $@

%.o : %.cu
	$(NVCC) -std=c++11 -arch=sm_35 -dc  -Xcompiler -fPIC $^ -o test.o
	$(NVCC) -std=c++11 -arch=sm_35 -Xcompiler -fPIC -dlink test.o -lcudadevrt -lcudart -o $@	

-include $(DEPS)



#prof.so: $(OBJECTS)
#	$(CXX) -shared $(LIB) $^ test.o -o $@
#%.o : %.cu
#	$(NVCC) $(NVCCFLAGS) -dc $^ -lcudadevrt -lcudart -o $@	

