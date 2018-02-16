#! /bin/bash

set -eou pipefail
set -x

PREFIX=$(readlink -f .)
NPROC=$(nproc)


# Install LibreSSL 2.6.4
# wget https://ftp.openbsd.org/pub/OpenBSD/LibreSSL/libressl-2.6.4.tar.gz -O libressl.tar.gz
# tar -xf libressl.tar.gz
# cd libressl-2.6.4 \
#   && ./configure --prefix="$PREFIX" \
#   && make -j$NPROC\
#   && make install \
#   && cd $PREFIX

# Install opentracing-cpp 1.2.0
# wget https://github.com/opentracing/opentracing-cpp/archive/v1.2.0.tar.gz -O opentracing-cpp.tar.gz
# tar -xf opentracing-cpp.tar.gz
# cd opentracing-cpp-1.2.0 \
#       && mkdir -p build \
#       && cd build \
#       && cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="$PREFIX" \
#       && make VERBOSE=1 \
#       && make install \
#       && cd $PREFIX


# Install zipkin-opentracing-cpp
# wget https://github.com/rnburn/zipkin-cpp-opentracing/archive/v0.2.0.tar.gz -O zipkin-opentracing-cpp.tar.gz
# tar -xf zipkin-opentracing-cpp.tar.gz
# cd zipkin-cpp-opentracing-0.2.0 \
#       && mkdir -p build \
#       && cd build \
#       && cmake -E env LDFLAGS="-L$PREFIX -lssl" \
#       cmake .. -DCMAKE_INSTALL_TYPE=Debug \
#                   -DCMAKE_INSTALL_PREFIX=$PREFIX \
#                   -DOPENTRACING_INCLUDE_DIR=$PREFIX/include \
#                   -DOPENTRACING_LIB=$PREFIX/lib/libopentracing.so \
#       && make VERBOSE=1 \
#       && make install \
#       && cd $PREFIX