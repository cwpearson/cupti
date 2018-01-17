#ifndef CPROF_MODEL_CUDA_KERNEL_RECORD_HPP
#define CPROF_MODEL_CUDA_KERNEL_RECORD_HPP

#include <chrono>
#include <string>

namespace cprof {
namespace model {
namespace cuda {

class KernelRecord {
  typedef std::chrono::high_resolution_clock::time_point time_point_t;

public:
  KernelRecord(const std::string &name) : name_(name) {}
  const std::string name() const noexcept { return name_; }

private:
  std::string name_;
  time_point_t start_;
  time_point_t end_;
  uint32_t cuptiCorrelationId_;
};

} // namespace cuda
} // namespace model
} // namespace cprof

#endif