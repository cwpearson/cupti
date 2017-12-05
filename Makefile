all: libcprof.so libprofiler.so

.PHONY: libcprof.so
libcprof.so:
	make -C cprof

.PHONY: libprofiler.so
libprofiler.so: libcprof.so
	make -C profiler
