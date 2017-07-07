#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <list>

#include "callbacks.hpp"

typedef cudaError_t (*cudaMallocFunc)(void **, size_t);
static cudaMallocFunc real_cudaMalloc = nullptr;

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  onceActivateCallbacks();

  if (real_cudaMalloc == nullptr) {
    real_cudaMalloc = (cudaMallocFunc)dlsym(RTLD_NEXT, "cudaMalloc");
  }
  assert(real_cudaMalloc && "Will the real cudaMalloc please stand up?");
  return real_cudaMalloc(devPtr, size);
}

typedef cudaError_t (*cudaMallocManagedFunc)(void **, size_t, unsigned int);
static cudaMallocManagedFunc real_cudaMallocManaged = nullptr;

extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                         unsigned int flags) {
  onceActivateCallbacks();

  if (real_cudaMallocManaged == nullptr) {
    real_cudaMallocManaged =
        (cudaMallocManagedFunc)dlsym(RTLD_NEXT, "cudaMallocManaged");
  }
  assert(real_cudaMallocManaged &&
         "Will the real cudaMallocManaged please stand up?");
  return real_cudaMallocManaged(devPtr, size, flags);
}

typedef cublasStatus_t (*cublasDgemvFunc)(cublasHandle_t, cublasOperation_t,
                                          int, int, const double *,
                                          const double *, int, const double *,
                                          int, const double *, double *, int);
static cublasDgemvFunc real_cublasDgemv = nullptr;

extern "C" cublasStatus_t cublasDgemv(cublasHandle_t handle,
                                      cublasOperation_t trans, int m, int n,
                                      const double *alpha, const double *A,
                                      int lda, const double *x, int incx,
                                      const double *beta, double *y, int incy) {
  printf("prof.so intercepted cublasDgemv call\n");

  if (real_cublasDgemv == nullptr) {
    real_cublasDgemv = (cublasDgemvFunc)dlsym(RTLD_NEXT, "cublasDgemv_v2");
  }
  assert(real_cublasDgemv && "Will the real cublasDgemv please stand up?");
  return real_cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                          incy);
}
// typedef cudaError_t
// (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
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
// typedef cudaError_t
// (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
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
