#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <list>
#include <cassert>
#include <dlfcn.h>

#include "callbacks.hpp"
#include "data.hpp"

typedef cudaError_t (*cudaMallocFunc)(void**,size_t);
static cudaMallocFunc real_cudaMalloc = NULL;

extern "C"
cudaError_t cudaMalloc(void **devPtr, size_t size) {
  lazyInitCallbacks();

  if (real_cudaMalloc == nullptr) {
    real_cudaMalloc = (cudaMallocFunc)dlsym(RTLD_NEXT,"cudaMalloc");
  }
  assert(real_cudaMalloc && "Will the real cudaMalloc please stand up?");
  return real_cudaMalloc(devPtr, size); 
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

/*
typedef void* (*mallocFunc)(size_t);
static mallocFunc real_malloc = nullptr;

void* malloc(size_t size) {
  if(!real_malloc) {
    real_malloc = (mallocFunc) dlsym(RTLD_NEXT, "malloc");
  }
  assert(real_malloc && "Will the real malloc please stand up");

  void *p = real_malloc(size);

  Value newValue;
  newValue.pos_ = (uintptr_t) p;
  newValue.size_ = size;

  Data::instance().values_.push_back(newValue);
  return p;
}
*/
