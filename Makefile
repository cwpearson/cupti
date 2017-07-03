TARGETS = prof.so

all: $(TARGETS)

OBJECTS = numa.o prof.o callbacks.o value.o values.o allocation.o allocations.o extent.o set_device.o hash.o

CXX=g++
CXXFLAGS= -std=c++11 -g -fno-omit-frame-pointer -Wall -Wextra -Wshadow -Wpedantic -fPIC
NVCC=nvcc
NVCCFLAGS = -std=c++11 -g -Xcompiler -fPIC,-fno-omit-frame-pointer -arch=sm_35
INC = -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB = -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -ldl -lnuma

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INC) $^ -c -o $@	

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -rdc=true $^ -c -o $@	

prof.so: $(OBJECTS)
	$(CXX) -shared $^ $(LIB) -o $@

clean:
	rm -f *.o cprof prof.so
