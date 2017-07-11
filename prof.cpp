#include <cassert>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <dlfcn.h>
#include <list>

#include "allocations.hpp"
#include "callbacks.hpp"
#include "thread.hpp"
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
  // printf("prof.cpp %d\n", get_thread_id());
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
  // printf("prof.cpp %d\n", get_thread_id());

  if (real_cudaMallocHost == nullptr) {
    real_cudaMallocHost =
        (cudaMallocHostFunc)dlsym(RTLD_NEXT, "cudaMallocHost");
  }
  assert(real_cudaMallocHost &&
         "Will the real cudaMallocHost please stand up?");
  return real_cudaMallocHost(ptr, size);
}

typedef cudaError_t (*cudaFreeHostFunc)(void *ptr);
static cudaFreeHostFunc real_cudaFreeHost = nullptr;
extern "C" cudaError_t cudaFreeHost(void *ptr) {
  onceActivateCallbacks();
  printf("prof.cpp %d\n", get_thread_id());

  if (real_cudaFreeHost == nullptr) {
    real_cudaFreeHost = (cudaFreeHostFunc)dlsym(RTLD_NEXT, "cudaFreeHost");
  }
  assert(real_cudaFreeHost && "Will the real cudaFreeHost please stand up?");
  return real_cudaFreeHost(ptr);
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
  Values::id_type aKey, xKey, yKey;
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
  printf("WARN: disabling CUPTI callbacks during cublasDgemv "
         "call\n");
  const cublasStatus_t ret = real_cublasDgemv(handle, trans, m, n, alpha, A,
                                              lda, x, incx, beta, y, incy);
  lazyActivateCallbacks();

  return ret;
}

typedef cublasStatus_t (*cublasSdotFunc)(cublasHandle_t handle, int n,
                                         const float *x, int incx,
                                         const float *y, int incy,
                                         float *result);
static cublasSdotFunc real_cublasSdot = nullptr;
extern "C" cublasStatus_t cublasSdot(cublasHandle_t handle, int n,
                                     const float *x, int incx, const float *y,
                                     int incy, float *result) {
  printf("prof.so intercepted cublasSdot call\n");

  if (real_cublasSdot == nullptr) {
    real_cublasSdot = (cublasSdotFunc)dlsym(RTLD_NEXT, "cublasSdot_v2");
  }
  assert(real_cublasSdot && "Will the real cublasSdot please stand up?");

  // record data, we know things about how this API works
  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type xId, yId;
  printf("Looking for x=%lu\n", (uintptr_t)x);
  std::tie(xId, std::ignore) =
      values.find_live((uintptr_t)x, AddressSpace::Cuda);
  assert(xId && "Couldn't find cublasSdot x argument value on device");
  std::tie(yId, std::ignore) =
      values.find_live((uintptr_t)y, AddressSpace::Cuda);
  assert(yId && "Couldn't find cublasSdot y argument value on device");

  // see if we can find an allocation for the result
  printf("Looking for allocation result=%lu\n", (uintptr_t)result);
  Allocations::id_type rAllocId;
  std::tie(rAllocId, std::ignore) = allocations.find_live(
      (uintptr_t)result, sizeof(float), AddressSpace(AddressSpace::Cuda));

  if (rAllocId == Allocations::noid) {
    printf("WARN: creating implicit allocation for cublasSdot result\n");
    AddressSpace AS = AddressSpace::Cuda;
    Memory AM = Memory(Memory::Unknown);
    auto pair = allocations.insert(std::shared_ptr<AllocationRecord>(
        new AllocationRecord((uintptr_t)result, sizeof(float), AS, AM,
                             AllocationRecord::PageType::Unknown)));
    assert(pair.second);
    rAllocId = pair.first->first;
  }
  printf("result allocId=%lu\n", rAllocId);
  // Make a new value
  Values::id_type rId;
  Values::value_type rVal;
  std::tie(rId, rVal) =
      values.new_value((uintptr_t)result, sizeof(float), rAllocId);
  rVal->add_depends_on(xId);

  lazyStopCallbacks();
  printf("WARN: disabling CUPTI callbacks during cublasSdot call\n");
  const cublasStatus_t ret =
      real_cublasSdot(handle, n, x, incx, y, incy, result);
  lazyActivateCallbacks();
  return ret;
}

typedef cublasStatus_t (*cublasSasumFunc)(cublasHandle_t, int, const float *,
                                          int, float *);
static cublasSasumFunc real_cublasSasum = nullptr;
extern "C" cublasStatus_t cublasSasum(cublasHandle_t handle, int n,
                                      const float *x, int incx, float *result) {
  printf("prof.so intercepted cublasSasum call\n");

  if (real_cublasSasum == nullptr) {
    real_cublasSasum = (cublasSasumFunc)dlsym(RTLD_NEXT, "cublasSasum_v2");
  }
  assert(real_cublasSasum && "Will the real cublasSasum please stand up?");

  // record data, we know things about how this API works
  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type xId;
  std::tie(xId, std::ignore) =
      values.find_live((uintptr_t)x, AddressSpace(AddressSpace::Cuda));
  assert(xId && "Couldn't find Sasum x argument value on device");

  // see if we can find an allocation for the result
  Allocations::id_type rAllocId;
  std::tie(rAllocId, std::ignore) = allocations.find_live(
      (uintptr_t)result, sizeof(float), AddressSpace::Cuda);

  if (!rAllocId) {
    // FIXME - can we do a better job with some parameters here
    AddressSpace AS(AddressSpace::Cuda);
    Memory AM(Memory::Unknown);
    std::tie(rAllocId, std::ignore) =
        allocations.new_allocation((uintptr_t)result, sizeof(float), AS, AM,
                                   AllocationRecord::PageType::Unknown);
    printf("WARN: new allocId=%lu for result=%lu\n", rAllocId,
           (uintptr_t)result);
  }
  assert(rAllocId && "If there is no allocation, we need to make one");

  // Make a new value
  Values::id_type rId;
  Values::value_type rVal;
  std::tie(rId, rVal) =
      values.new_value((uintptr_t)result, sizeof(float), rAllocId);
  rVal->add_depends_on(xId);

  lazyStopCallbacks();
  printf("WARN: disabling CUPTI callbacks during cublasSasum call\n");
  const cublasStatus_t ret = real_cublasSasum(handle, n, x, incx, result);
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
  onceActivateCallbacks();
  printf("prof.so intercepted cublasCreate call\n");

  if (real_cublasCreate == nullptr) {
    real_cublasCreate = (cublasCreateFunc)dlsym(RTLD_NEXT, "cublasCreate_v2");
  }
  assert(real_cublasCreate && "Will the real cublasCreate please stand up?");

  printf("WARN: disabling CUPTI callbacks during cublasCreate call\n");
  lazyStopCallbacks();
  const cublasStatus_t ret = real_cublasCreate(handle);
  lazyActivateCallbacks();
  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionForwardFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);
static cudnnConvolutionForwardFunc real_cudnnConvolutionForward = nullptr;
extern "C" cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {

  /*
  y = f(x,w,alpha,beta, workspace)
  */
  onceActivateCallbacks();
  printf("prof.so intercepted cudnnConvolutionForward call\n");

  if (real_cudnnConvolutionForward == nullptr) {
    real_cudnnConvolutionForward = (cudnnConvolutionForwardFunc)dlsym(
        RTLD_NEXT, "cudnnConvolutionForward");
  }
  assert(real_cublasCreate &&
         "Will the real cudnnConvolutionForward please stand up?");

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Find input values
  printf("Looking for x=%lu, w=%lu, workSpace=%lu\n", (uintptr_t)x,
         (uintptr_t)w, (uintptr_t)workSpace);
  Values::id_type xId, wId, workSpaceId;
  AddressSpace devOrUnified = AddressSpace::Cuda;
  std::tie(xId, std::ignore) = values.find_live((uintptr_t)x, devOrUnified);
  std::tie(wId, std::ignore) = values.find_live((uintptr_t)w, devOrUnified);
  std::tie(workSpaceId, std::ignore) =
      values.find_live((uintptr_t)workSpace, devOrUnified);
  assert(xId && wId && workSpaceId &&
         "Couldn't find cudnnConvolutionForward argument value on device");

  // See if there is an existing output value to take info from
  Values::id_type yId;
  Values::value_type yVal;
  std::tie(yId, yVal) = values.find_live((uintptr_t)w, devOrUnified);
  if (yId == Values::noid) {
    Allocations::id_type yAllocId;
    std::tie(yAllocId, std::ignore) =
        allocations.find_live((uintptr_t)y, AddressSpace::Cuda);
    assert(yAllocId != Allocations::noid && "Couldn't find allocation");
    std::tie(yId, yVal) = values.new_value(uintptr_t(y), 0, yAllocId); // FIXME
    printf("WARN: create new y=%lu value=%lu\n", (uintptr_t)y, yId);
  }
  yVal->add_depends_on(xId);
  yVal->add_depends_on(wId);
  yVal->add_depends_on(workSpaceId);
  printf("[cudnnConvolutionForward] %lu deps on %lu %lu %lu\n", yId, xId, wId,
         workSpaceId);

  printf(
      "WARN: disabling CUPTI callbacks during cudnnForwardConvolution call\n");
  lazyStopCallbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionForward(
      handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, yDesc, y);
  lazyActivateCallbacks();
  return ret;
}