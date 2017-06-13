TARGETS = query vec prof.so

all: $(TARGETS)

NVCC_FLAGS=--cudart=shared
INC = -I/usr/local/cuda/include -I/usr/local/cuda/extras/CUPTI/include
LIB = -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -ldl

query: deviceQuery.cpp
	nvcc $(NVCC_FLAGS) $^ -o $@

prof.so: prof.cpp callbacks.cpp
	g++ -Wall -Wextra -std=c++11 $(INC) $(LIB) -shared -fPIC $^ -o $@

vec: vectorAdd.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f query cprof vec prof.so
