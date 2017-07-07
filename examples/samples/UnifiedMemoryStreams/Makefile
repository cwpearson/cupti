TARGET = ums
all: $(TARGET)

NVCC_FLAGS=--cudart=shared -I../common/inc -lgomp -lcublas

$(TARGET): UnifiedMemoryStreams.cu
	nvcc $(NVCC_FLAGS) $^ -o $@

clean:
	rm -f *.o $(TARGET)
