#! /bin/bash

## Adjust these variables to match your installation
# CUPTI should be in the LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/software/opentracing-cpp/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/software/zipkin-cpp-opentracing/lib:$LD_LIBRARY_PATH"
# where to look for prof.so
export CPROF_ROOT="$HOME/repos/cupti"



## Control some profiling parameters.
# Check env.sh for more info
export CPROF_OUT="output.cprof"



## Run the provided program. For example
#   ./env.sh examples/samples/vectorAdd/vec
LD_PRELOAD="$LD_PRELOAD:$CPROF_ROOT/lib/libcprof.so" gdb $@

# set environment LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/home/pearson/software/opentracing-cpp/lib:/home/pearson/software/zipkin-cpp-opentracing/lib