#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <list>

#include "callbacks.hpp"
#include "values.hpp"

typedef cudaError_t (*cudaGetDeviceCountFunc)(int *);
static cudaGetDeviceCountFunc real_cudaGetDeviceCount = nullptr;

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  onceActivateCallbacks();

  if (real_cudaGetDeviceCount == nullptr) {
    real_cudaGetDeviceCount =
        (cudaGetDeviceCountFunc)dlsym(RTLD_NEXT, "cudaGetDeviceCount");
  }
  assert(real_cudaGetDeviceCount &&
         "Will the real cudaGetDeviceCount please stand up?");
  return real_cudaGetDeviceCount(count);
}

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

typedef cudaError_t (*cudaMallocHostFunc)(void **ptr, size_t);
static cudaMallocHostFunc real_cudaMallocHost = nullptr;

extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
  onceActivateCallbacks();

  if (real_cudaMallocHost == nullptr) {
    real_cudaMallocHost =
        (cudaMallocHostFunc)dlsym(RTLD_NEXT, "cudaMallocHost");
  }
  assert(real_cudaMallocHost &&
         "Will the real cudaMallocHost please stand up?");
  return real_cudaMallocHost(ptr, size);
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

  // record data, we know things about how this API works
  auto &values = Values::instance();

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::key_type aKey, xKey, yKey;
  Values::value_type aVal, xVal, yVal;
  std::tie(aKey, aVal) = values.find_live_device((uintptr_t)A, 1);
  std::tie(xKey, xVal) = values.find_live_device((uintptr_t)x, 1);
  std::tie(yKey, yVal) = values.find_live_device((uintptr_t)y, 1);

  assert(aKey && xKey && yKey &&
         "Couldn't find Dgemv argument value on device");

  // FIXME: could use these to do better on dependences
  printf("WARN: not handling some values (A, alpha, beta)\n");

  const auto newValue =
      std::shared_ptr<Value>(new Value(*yVal)); // duplicate the value
  values.insert(newValue);
  newValue->add_depends_on(aKey);
  newValue->add_depends_on(xKey);
  newValue->add_depends_on(yKey);

  lazyStopCallbacks();
  printf("WARN: disabling CUPTI callbacks during cublasDgemv call\n");
  const cublasStatus_t ret = real_cublasDgemv(handle, trans, m, n, alpha, A,
                                              lda, x, incx, beta, y, incy);
  lazyActivateCallbacks();

  return ret;
}

typedef cublasStatus_t (*cublasDestroyFunc)(cublasHandle_t handle);
static cublasDestroyFunc real_cublasDestroy = nullptr;
extern "C" cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  printf("prof.so intercepted cublasDestroy call\n");

  if (real_cublasDestroy == nullptr) {
    real_cublasDestroy =
        (cublasDestroyFunc)dlsym(RTLD_NEXT, "cublasDestroy_v2");
  }
  assert(real_cublasDestroy && "Will the real cublasDestroy please stand up?");

  lazyStopCallbacks();
  printf("WARN: disabling CUPTI callbacks during cublasDestroy call\n");
  const cublasStatus_t ret = real_cublasDestroy(handle);
  lazyActivateCallbacks();
  return ret;
}

typedef cublasStatus_t (*cublasCreateFunc)(cublasHandle_t *handle);
static cublasCreateFunc real_cublasCreate = nullptr;
extern "C" cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  printf("prof.so intercepted cublasCreate call\n");

  if (real_cublasCreate == nullptr) {
    real_cublasCreate = (cublasCreateFunc)dlsym(RTLD_NEXT, "cublasCreate_v2");
  }
  assert(real_cublasCreate && "Will the real cublasCreate please stand up?");

  lazyStopCallbacks();
  printf("WARN: disabling CUPTI callbacks during cublasCreate call\n");
  const cublasStatus_t ret = real_cublasCreate(handle);
  lazyActivateCallbacks();
  return ret;
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
