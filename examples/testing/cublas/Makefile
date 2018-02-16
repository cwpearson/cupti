TARGETS = main
all: $(TARGETS)

NVCC_FLAGS=--cudart=shared -I../common/inc 

main: cublas_test.cu
	nvcc $(NVCC_FLAGS) -lcublas $^ -o $@

clean:
	rm -f *.o $(TARGETS)
