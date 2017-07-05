#! /bin/bash

CPROF_OUT="cprof.txt"

LD_PRELOAD="$LD_PRELOAD:/usr/local/cuda/extras/CUPTI/lib64/libcupti.so"
LD_PRELOAD="$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libnuma.so"
LD_PRELOAD="$LD_PRELOAD:/usr/lib/powerpc64le-linux-gnu/libnuma.so"
LD_PRELOAD="$LD_PRELOAD:$PWD/prof.so" $@
