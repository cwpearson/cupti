#include "cprof/model/cuda/device.hpp"

using namespace cprof::model::cuda;

// Device::Device()
//     : id_(-1), unifiedAddressing_(0), canMapHostMemory_(0),
//       pageableMemoryAccess_(0), concurrentManagedAccess_(0), major_(0),
//       minor_(0) {}

Device::Device(const cudaDeviceProp &prop, int id)
    : id_(id), unifiedAddressing_(prop.unifiedAddressing),
      canMapHostMemory_(prop.canMapHostMemory),
      pageableMemoryAccess_(prop.pageableMemoryAccess),
      concurrentManagedAccess_(prop.concurrentManagedAccess),
      major_(prop.major), minor_(prop.minor) {}