#ifndef MEMORYCOPYKIND_HPP
#define MEMORYCOPYKIND_HPP

#include <cassert>
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
      logging::err() << "Unhandled cudaMemcpyKind " << kind << std::endl;
      assert(0 && "Unsupported cudaMemcpy kind");
    }
  }

  std::string str() const {
    switch (type_) {
    case Type::CudaHostToDevice:
      return "cudaMemcpyHostToDevice";
    case Type::CudaDeviceToHost:
      return "cudaMemcpyDeviceToHost";
    case Type::CudaHostToHost:
      return "cudaMemcpyHostToHost";
    case Type::CudaDeviceToDevice:
      return "cudaMemcpyDeviceToDevice";
    case Type::CudaDefault:
      return "cudaMemcpyDefault";
    case Type::CudaPeer:
      return "cudaMemcpyPeer";
    default:
      assert(0 && "Unhandled MemoryCopyKind");
    }
  }

  bool operator==(const MemoryCopyKind &rhs) const {
    return type_ == rhs.type_;
  }

  static MemoryCopyKind CudaHostToHost() {
    return MemoryCopyKind(Type::CudaHostToHost);
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