# CUPTI

| master | develop |
|--------|---------|
| coming soon... | [![Build Status](https://travis-ci.org/cwpearson/cupti.svg?branch=develop)](https://travis-ci.org/cwpearson/cupti)

## Setup

Install some dependencies

    sudo apt install libnuma-dev libboost-all-dev

Install CUDA and CUDNN.

Create a `Makfile.config`

    cp Makefile.config.example Makefile.config

Edit that makefile to match your system setup.

Make sure the CUPTI library is in your `LD_LIBRARY_PATH`. For example:

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64"

Modify `env.sh` to point `CPROF_ROOT` to wherever you checked out cupti. For example

    export CPROF_ROOT="/home/pearson/cupti"

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