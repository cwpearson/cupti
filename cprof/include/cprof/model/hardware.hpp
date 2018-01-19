#ifndef CPROF_MODEL_SYSTEM_HPP
#define CPROF_MODEL_SYSTEM_HPP

#include <vector>

#include "cprof/address_space.hpp"
#include "cprof/model/cuda/device.hpp"

namespace cprof {
namespace model {

/*! \brief Represents information about the physical system
 *
 * We need to know some information about the profiled system in order
 * to fully understand the semantics of various APIs.
 */
class Hardware {
  std::vector<cuda::Device> cudaDevices_;

public:
  const cuda::Device &cuda_device(size_t i) { return cudaDevices_[i]; }
  void get_device_properties();

  /*! \brief Address space a device participates in
   */
  const AddressSpace address_space(const int dev) {
    if (cudaDevices_[dev].unifiedAddressing_) {
      return AddressSpace::CudaUVA();
    }
    return AddressSpace::CudaDevice(dev);
  }
};

} // namespace model
} // namespace cprof

#endif