#! /bin/bash
set -eou pipefail

## Adjust these variables to match your installation
# CUPTI should be in the LD_LIBRARY_PATH

if [ -z "${OPENTRACING_LIB+xxx}" ]; then 
  export OPENTRACING_LIB="$HOME/software/opentracing-cpp/lib";
fi
if [ -z "${ZIPKIN_LIB+xxx}" ]; then 
  export ZIPKIN_LIB="$HOME/software/zipkin-cpp-opentracing/lib";
fi

export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$OPENTRACING_LIB:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$ZIPKIN_LIB:$LD_LIBRARY_PATH"

# where to look for prof.so
if [ -z "${CPROF_ROOT+xxx}" ]; then 
  export CPROF_ROOT="$HOME/repos/cupti"; # not set at all
fi

## Control some profiling parameters.

# default output file
export CPROF_OUT="output.cprof"
#export CPROF_ERR="err.cprof"


# endpoint for tracing
export CPROF_ENABLE_ZIPKIN=0
export CPROF_ZIPKIN_HOST=34.215.126.137
export CPROF_ZIPKIN_PORT=16686

## Run the provided program. For example
#   ./env.sh examples/samples/vectorAdd/vec

if [ -z "${LD_PRELOAD+xxx}" ]; then 
  LD_PRELOAD="$CPROF_ROOT/lib/libcprof.so" gdb $@; # unset
else
  echo "Error: LD_PRELOAD is set"
fi
