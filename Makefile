TARGETS = prof.so

all: $(TARGETS)

OBJECTS = numa.o prof.o callbacks.o value.o values.o allocation.o allocations.o extent.o set_device.o hash.o

CXX=g++
CXXFLAGS= -std=c++11 -g -fno-omit-frame-pointer -Wall -Wextra -Wshadow -Wpedantic -fPIC
NVCC=nvcc
NVCCFLAGS= -std=c++11 -g -arch=sm_35 -Xcompiler -Wall,-Wextra,-fPIC,-fno-omit-frame-pointer
INC = -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB = -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -L/usr/local/cuda/lib64 -lcudart -lcudadevrt -ldl -lnuma

%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INC) $^ -c -o $@	

#%.o : %.cu
#	$(NVCC) $(NVCCFLAGS) -dc $^ -lcudadevrt -lcudart -o $@	

%.o : %.cu
	$(NVCC) -std=c++11 -arch=sm_52 -dc  -Xcompiler -fPIC $^ -o test.o
	$(NVCC) -std=c++11 -arch=sm_52 -Xcompiler -fPIC -dlink test.o -lcudadevrt -lcudart -o $@	

prof.so: $(OBJECTS)
	$(CXX) -shared $(LIB) $^ test.o -o $@

clean:
	rm -f *.o cprof prof.so
