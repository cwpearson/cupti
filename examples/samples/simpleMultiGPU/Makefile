TARGETS = mgpu
all: $(TARGETS)

NVCC_FLAGS=--cudart=shared -I../common/inc

mgpu: simpleMultiGPU.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o $(TARGETS)
