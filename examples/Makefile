TARGETS = vec

all: $(TARGETS)

NVCC_FLAGS=--cudart=shared

vec: vectorAdd.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o vec
