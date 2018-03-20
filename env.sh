#! /bin/bash
set -eou pipefail

# the time of the profile
NOW=`date +%Y%m%d-%H%M%S`

# where to look for cprof/profiler
if [ -z "${CPROF_ROOT+xxx}" ]; then 
  export CPROF_ROOT="$HOME/repos/cupti"; # not set at all
fi

# Check that libcprof.so exists
LIBCPROF="$CPROF_ROOT/cprof/lib/libcprof.so"
if [ ! -f "$LIBCPROF" ]; then
    echo "$LIBCPROF" "not found! try"
    echo "make -C $CPROF_ROOT/cprof"
    exit -1
fi

# Check that libprofiler.so exists
LIBPROFILER="$CPROF_ROOT/profiler/lib/libprofiler.so"
if [ ! -f "$LIBPROFILER" ]; then
    echo "$LIBPROFILER" "not found! try"
    echo "make -C $CPROF_ROOT/profiler"
    exit -1
fi


## Control some profiling parameters.

# default output file
export CPROF_OUT="$NOW"_output.cprof
#export CPROF_ERR="err.cprof"

export CPROF_ENABLE_ZIPKIN=0
export CPROF_ZIPKIN_HOST=34.215.126.137
export CPROF_ZIPKIN_PORT=9411
export CPROF_CHROME_TRACING=events_"$NOW".json
export CPROF_CUPTI_DEVICE_BUFFER_SIZE=1024
export CPROF_MODE=full
export CPROF_QUIET=0

## Run the provided program. For example
#   ./env.sh examples/samples/vectorAdd/vec

if [ -z "${LD_PRELOAD+xxx}" ]; then 
  LD_PRELOAD="$CPROF_ROOT/profiler/lib/libprofiler.so" $@; # unset
else
  echo "Error: LD_PRELOAD is set before profile:"
  echo "\tLD_PRELOAD=$LD_PRELOAD"
fi
