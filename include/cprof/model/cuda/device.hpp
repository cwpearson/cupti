#ifndef CPROF_MODEL_CUDA_DEVICE_HPP
#define CPROF_MODEL_CUDA_DEVICE_HPP

#include <string>

#include <cuda_runtime.h>

namespace cprof {
namespace model {
namespace cuda {

class Device {
public:
  Device(const cudaDeviceProp &prop);

public:
  std::string name_;            ///< cudaDeviceProp.name
  int id_;                      ///< cuda device id
  int unifiedAddressing_;       ///< cudaDeviceProp.unifiedAddressing
  int canMapHostMemory_;        ///< cudaDeviceProp.canMapHostMemory
  int pageableMemoryAccess_;    ///< cudaDeviceProp.pageableMemoryAccess
  int concurrentManagedAccess_; ///< cudaDeviceProp.concurrentManagedAccess
  int minor_;                   ///< cudaDeviceProp.minor
  int major_;                   ///< cudaDeviceProp.major
};

} // namespace cuda
} // namespace model
} // namespace cprof
#endif