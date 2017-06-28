TARGETS = query vec prof.so

all: $(TARGETS)

NVCC_FLAGS=--cudart=shared
INC = -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB = -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -ldl

query: deviceQuery.cpp
	nvcc $(NVCC_FLAGS) $^ -o $@

%.o : %.cpp
	g++ -g -Wall -Wextra -std=c++11 -fPIC $(INC) $^ -c -o $@	

prof.so: prof.o callbacks.o value.o values.o allocation.o allocations.o extent.o
	g++ -g -Wall -Wextra -std=c++11 -shared -fPIC $^ $(LIB) -o $@

vec: vectorAdd.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o query cprof vec prof.so
