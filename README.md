# CUPTI

## Setup

Install some dependencies

    sudo apt install libnuma-dev libboost-all-dev

Install CUDA and CUDNN.

    ...

Modify `env.sh` to point to the right libraries.

Build the profiling library (`prof.so`).

    make

## Run on a CUDA application

Make sure your CUDA application is not statically-linked, which is the default when you are building your own CUDA code.

This will record data by appending to an output.cprof file, so usually remove that file first. `./env.sh` sets up the `LD_PRELOAD` environment and invokes your app.

    rm -f output.cprof
    ./env.sh <your app>

Do something with the result:

    cprof2<something>.py

Other info

`env.sh` sets `LD_PRELOAD` to load the profiling library and its dependences.