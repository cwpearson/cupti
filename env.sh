#! /bin/bash

CPROF_OUT="cprof.txt"

export LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

LD_PRELOAD="$LD_PRELOAD:$PWD/prof.so" $@
