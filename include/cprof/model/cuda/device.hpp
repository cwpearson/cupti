#ifndef CPROF_MODEL_CUDA_DEVICE_HPP
#define CPROF_MODEL_CUDA_DEVICE_HPP

#include <string>

#include <cuda_runtime_api.h>

namespace cprof {
namespace model {
namespace cuda {

class Device {
public:
  Device(const cudaDeviceProp &prop, int id);

public:
  std::string name_;            ///< cudaDeviceProp.name
  int id_;                      ///< cuda device id
  int unifiedAddressing_;       ///< cudaDeviceProp.unifiedAddressing
  int canMapHostMemory_;        ///< cudaDeviceProp.canMapHostMemory
  int pageableMemoryAccess_;    ///< cudaDeviceProp.pageableMemoryAccess
  int concurrentManagedAccess_; ///< cudaDeviceProp.concurrentManagedAccess
  int major_;                   ///< cudaDeviceProp.major
  int minor_;                   ///< cudaDeviceProp.minor
};

} // namespace cuda
} // namespace model
} // namespace cprof
#endif