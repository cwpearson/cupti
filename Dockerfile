FROM nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04
LABEL maintainer "carl.w.pearson@gmail.com"

ENV CUDA_ROOT /usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    libnuma-dev \
    libboost-all-dev \
    libcurl4-openssl-dev \ 
    cmake  \
    cppcheck \
    git \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install opentracing-cpp
RUN git clone https://github.com/opentracing/opentracing-cpp.git
RUN mkdir opentracing-cpp/build && \
    cd opentracing-cpp/build && \
    cmake ..
RUN make install -C opentracing-cpp/build
RUN rm -rf opentracing-cpp
ENV OPENTRACING_LIB /usr/local/lib

#install zipkin-cpp-opentracing
RUN git clone https://github.com/rnburn/zipkin-cpp-opentracing.git
RUN mkdir zipkin-cpp-opentracing/build \
    && cd zipkin-cpp-opentracing/build \
    && cmake ..
RUN make install -C zipkin-cpp-opentracing/build
ENV ZIPKIN_LIB /usr/local/lib

COPY . /cprof
COPY env.sh /bin/env.sh
WORKDIR /cprof

# Create makefile
RUN echo CUDA_ROOT=$CUDA_ROOT >> Makefile.config \
    && echo BUILD_TYPE=Debug >> Makefile.config

RUN nice -n20 make -j`nproc`

ENV CPROF_ROOT /cprof
ENTRYPOINT ["env.sh"]