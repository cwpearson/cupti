#include "cudnn_util.hpp"

#include <cassert>

#include "profiler.hpp"

size_t tensorSize(const cudnnTensorDescriptor_t tensorDesc) {
  size_t size;
  CUDNN_CHECK(cudnnGetTensorSizeInBytes(tensorDesc, &size), profiler::err());
  return size;
}