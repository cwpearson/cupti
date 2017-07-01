TARGETS = query

all: $(TARGETS)

NVCC_FLAGS=--cudart=shared

query: deviceQuery.cpp
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o query 
