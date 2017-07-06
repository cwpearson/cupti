# CUPTI

Install some dependencies

    sudo apt install libnuma-dev libboost-all-dev graphviz

Build the profiling library:

    make

Run on a CUDA application:

    ./env.sh <your app> && ./draw.py cprof.txt

Other info

`env.sh` sets `LD_PRELOAD` to load the profiling library and its dependences.

`draw.py` translates `cprof.txt` into `cprof.dot` and then `cprof.pdf` for viewing.