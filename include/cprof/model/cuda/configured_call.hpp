#ifndef CPROF_MODEL_CUDA_CONFIGURED_CALL_HPP
#define CPROF_MODEL_CUDA_CONFIGURED_CALL_HPP

#include <vector>

#include <cuda_runtime.h>

class ConfiguredCall {
public:
  ConfiguredCall() : valid_(false) {}
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  cudaStream_t stream_;
  std::vector<uintptr_t> args_;
  bool valid_;
};

#endif