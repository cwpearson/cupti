#! /bin/bash

export LD_PRELOAD="$LD_PRELOAD:/usr/local/cuda/extras/CUPTI/lib64/libcupti.so"
export LD_PRELOAD="$LD_PRELOAD:$PWD/prof.so"

exec bash
