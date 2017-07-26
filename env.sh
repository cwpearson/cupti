#! /bin/bash

export CPROF_OUT="output.cprof"

export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

LD_PRELOAD="$LD_PRELOAD:$PWD/prof.so" $@
