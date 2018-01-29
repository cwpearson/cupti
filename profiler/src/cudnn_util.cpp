#include "cudnn_util.hpp"

#include <cassert>

#include "profiler.hpp"

size_t cudnnDataTypeSize(const cudnnDataType_t type) {
  switch (type) {
  case CUDNN_DATA_FLOAT:
    return sizeof(float);
  case CUDNN_DATA_DOUBLE:
    return sizeof(double);
  case CUDNN_DATA_HALF:
    return 2;
  case CUDNN_DATA_INT8:
    return 1;
  case CUDNN_DATA_INT32:
    return 4;
  case CUDNN_DATA_INT8x4:
    return 4;
  default:
    assert(0 && "Unexpected cudnnDataType_t");
  }
}

size_t cudnnTensor4dSize(const cudnnTensorDescriptor_t tensorDesc) {
  int n, c, h, w, nStride, cStride, hStride, wStride;
  cudnnDataType_t dataType;
  CUDNN_CHECK(cudnnGetTensor4dDescriptor(tensorDesc, &dataType, &n, &c, &h, &w,
                                         &nStride, &cStride, &hStride,
                                         &wStride),
              profiler::err());

  return cudnnDataTypeSize(dataType) * 1;
}