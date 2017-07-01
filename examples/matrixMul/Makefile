TARGETS = mm

all: $(TARGETS)

NVCC_FLAGS=--cudart=shared

mm: matrixMul.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o $(TARGETS)
