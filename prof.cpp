#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <list>

#include <cuda.h>
#include <cuda_runtime.h>

#include "allocations.hpp"
#include "callbacks.hpp"
#include "driver_state.hpp"
#include "thread.hpp"
#include "values.hpp"

typedef cudaError_t (*cudaGetDeviceCountFunc)(int *);
extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  static cudaGetDeviceCountFunc real_cudaGetDeviceCount = nullptr;
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
extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  static cudaMallocFunc real_cudaMalloc = nullptr;
  onceActivateCallbacks();
  // printf("prof.cpp %d\n", get_thread_id());
  if (real_cudaMalloc == nullptr) {
    real_cudaMalloc = (cudaMallocFunc)dlsym(RTLD_NEXT, "cudaMalloc");
  }
  assert(real_cudaMalloc && "Will the real cudaMalloc please stand up?");
  return real_cudaMalloc(devPtr, size);
}

typedef cudaError_t (*cudaMallocManagedFunc)(void **, size_t, unsigned int);
extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                         unsigned int flags) {
  static cudaMallocManagedFunc real_cudaMallocManaged = nullptr;
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
extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
  static cudaMallocHostFunc real_cudaMallocHost = nullptr;
  onceActivateCallbacks();

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

