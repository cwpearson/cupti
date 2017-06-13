#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <list>
#include <cassert>
#include <dlfcn.h>

#include "callbacks.hpp"

extern "C"
cudaError_t cudaMalloc	(void **devPtr, size_t size) {
  fprintf(stderr, "[cudaMalloc] called\n");

  lazyInitCallbacks();

  return cudaSuccess; 
}

typedef cudaError_t (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
static cudaConfigureCall_t realCudaConfigureCall = NULL;

typedef struct {
  dim3 gridDim;
  dim3 blockDim;
  int counter;
  std::list<void *> args;
} kernelInfo_t;

kernelInfo_t &kernelInfo() {
  static kernelInfo_t _kernelInfo;
  return _kernelInfo;
}

extern "C"
cudaError_t cudaConfigureCall(
  dim3 gridDim,
  dim3 blockDim,
  size_t sharedMem,
  cudaStream_t stream) {
  fprintf(stderr, "cudaConfigureCall\n");
  assert(kernelInfo().counter == 0 && "Multiple cudaConfigureCalls before cudaLaunch?");
  kernelInfo().gridDim = gridDim;
  kernelInfo().blockDim = blockDim;
  kernelInfo().counter++;
  if (realCudaConfigureCall == NULL)
    realCudaConfigureCall = (cudaConfigureCall_t)dlsym(RTLD_NEXT,"cudaConfigureCall");
  assert(realCudaConfigureCall != NULL && "cudaConfigureCall is null");
  return realCudaConfigureCall(gridDim,blockDim,sharedMem,stream);
}
