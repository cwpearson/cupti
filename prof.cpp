#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <list>

#include "callbacks.hpp"

typedef cudaError_t (*cudaMallocFunc)(void **, size_t);
static cudaMallocFunc real_cudaMalloc = NULL;

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  onceActivateCallbacks();

  if (real_cudaMalloc == nullptr) {
    real_cudaMalloc = (cudaMallocFunc)dlsym(RTLD_NEXT, "cudaMalloc");
  }
  assert(real_cudaMalloc && "Will the real cudaMalloc please stand up?");
  return real_cudaMalloc(devPtr, size);
}

// typedef cudaError_t (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
// static cudaConfigureCall_t realCudaConfigureCall = NULL;

/*
typedef CUresult (*cuInitFunc)(unsigned int);
static cuInitFunc real_cuInit = nullptr;

extern "C"
CUresult cuInit(unsigned int Flags) {
  printf("intercepted cuInit\n");
  lazyActivateCallbacks();

  if (real_cuInit == NULL) {
    real_cuInit = (cuInitFunc)dlsym(RTLD_NEXT,"cuInit");
  }
  assert(real_cuInit && "Will the real cuInit please stand up.");
  return real_cuInit(Flags);
}
*/
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
