all: query cprof

query: deviceQuery.cpp
	nvcc $^ -o $@

cprof: cprof.cpp
	nvcc -std=c++11 -I/usr/local/cuda/extras/CUPTI/include -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti -lcuda -lcudart $^ -o $@

clean:
	rm -f query cprof
