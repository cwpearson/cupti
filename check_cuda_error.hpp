#ifndef CHECK_CUDA_ERROR_HPP
#define CHECK_CUDA_ERROR_HPP

#include <cstdio>
#include <cuda.h>

#define CHECK_CUDA_ERROR(err)                                                  \
  if (err != cudaSuccess) {                                                    \
    printf("%s:%d: error %d for CUDA Driver API function\n", __FILE__,         \
           __LINE__, err);                                                     \
    exit(-1);                                                                  \
  }

#endif