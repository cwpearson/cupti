#ifndef UTIL_CUDA_HPP
#define UTIL_CUDA_HPP

#include <cstdlib>
#include <cuda_runtime.h>
#include <ostream>

#define CUDA_CHECK(ans, err)                                                   \
  { gpuAssert((ans), __FILE__, __LINE__, (err)); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      std::ostream &err, bool abort = true) {
  if (code != cudaSuccess) {
    err << "CUDA_CHECK: " << cudaGetErrorString(code) << " " << file << " "
        << line << std::endl;
    if (abort)
      exit(code);
  }
}

#endif