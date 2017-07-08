#ifndef MEMORYCOPYKIND_HPP
#define MEMORYCOPYKIND_HPP

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

class MemoryCopyKind {
private:
  enum class Type {
    CudaHostToHost,
    CudaHostToDevice,
    CudaDeviceToHost,
    CudaDeviceToDevice,
    CudaDefault,
    CudaPeer
  };
  Type type_;

public:
  MemoryCopyKind(MemoryCopyKind::Type type) : type_(type) {}
  MemoryCopyKind(const cudaMemcpyKind kind) {
    if (cudaMemcpyHostToHost == kind) {
      type_ = Type::CudaHostToHost;
    } else if (cudaMemcpyHostToDevice == kind) {
      type_ = Type::CudaHostToDevice;
    } else if (cudaMemcpyDeviceToHost == kind) {
      type_ = Type::CudaDeviceToHost;
    } else if (cudaMemcpyDeviceToDevice == kind) {
      type_ = Type::CudaDeviceToDevice;
    } else if (cudaMemcpyDefault == kind) {
      type_ = Type::CudaDefault;
    } else {
      printf("Unhandled cudaMemcpyKind %d\n", kind);
      assert(0 && "Unsupported cudaMemcpy kind");
    }
  }

  bool operator==(const MemoryCopyKind &rhs) const {
    return type_ == rhs.type_;
  }

  static MemoryCopyKind CudaPeer() { return MemoryCopyKind(Type::CudaPeer); }
  static MemoryCopyKind CudaHostToDevice() {
    return MemoryCopyKind(Type::CudaHostToDevice);
  }
  static MemoryCopyKind CudaDeviceToHost() {
    return MemoryCopyKind(Type::CudaDeviceToHost);
  }
  static MemoryCopyKind CudaDeviceToDevice() {
    return MemoryCopyKind(Type::CudaDeviceToDevice);
  }
  static MemoryCopyKind CudaDefault() {
    return MemoryCopyKind(Type::CudaDefault);
  }
};

#endif