// #include <cassert>
// #include <cuda_runtime.h>
// #include <dlfcn.h>

/*
typedef cudaError_t (*cudaFreeHostFunc)(void *ptr);
extern "C" cudaError_t cudaFreeHost(void *ptr) {
  SAME_LD_PRELOAD_BOILERPLATE(cudaFreeHost);
  return real_cudaFreeHost(ptr);
}


typedef cudaError_t (*cudaGetDeviceCountFunc)(int *);
extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  SAME_LD_PRELOAD_BOILERPLATE(cudaGetDeviceCount);
  return real_cudaGetDeviceCount(count);
}

typedef cudaError_t (*cudaMallocFunc)(void **, size_t);
extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  SAME_LD_PRELOAD_BOILERPLATE(cudaMalloc);
  return real_cudaMalloc(devPtr, size);
}

typedef cudaError_t (*cudaMallocHostFunc)(void **ptr, size_t);
extern "C" cudaError_t cudaMallocHost(void **ptr, size_t size) {
  SAME_LD_PRELOAD_BOILERPLATE(cudaMallocHost);
  return real_cudaMallocHost(ptr, size);
}

typedef cudaError_t (*cudaMallocManagedFunc)(void **, size_t, unsigned int);
extern "C" cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                                         unsigned int flags) {
  SAME_LD_PRELOAD_BOILERPLATE(cudaMallocManaged);
  return real_cudaMallocManaged(devPtr, size, flags);
}

typedef cudaError_t (*cudaSetDeviceFunc)(int device);
extern "C" cudaError_t cudaSetDevice(int device) {
  SAME_LD_PRELOAD_BOILERPLATE(cudaSetDevice);
  return real_cudaSetDevice(device);
}

*/