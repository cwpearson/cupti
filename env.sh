#! /bin/bash

## Adjust these variables to match your installation
# CUPTI should be in the LD_LIBRARY_PATH
export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
# where to look for prof.so
export CPROF_ROOT="$HOME/cupti"



## Control some profiling parameters.
# Check env.sh for more info
export CPROF_OUT="output.cprof"



## Run the provided program. For example
#   ./env.sh examples/samples/vectorAdd/vec
LD_PRELOAD="$LD_PRELOAD:$CPROF_ROOT/prof.so" $@
