#ifndef CPROF_MODEL_CUDA_KERNEL_RECORD_HPP
#define CPROF_MODEL_CUDA_KERNEL_RECORD_HPP

#include <chrono>
#include <string>

namespace cprof {
namespace model {
namespace cuda {

class KernelRecord {

public:
  KernelRecord(const std::string &name, uint32_t cuptiCorrelationId)
      : start_(0), end_(0), name_(name),
        cuptiCorrelationId_(cuptiCorrelationId) {}
  const std::string name() const noexcept { return name_; }
  uint64_t start_;
  uint64_t end_;

private:
  std::string name_;
  uint32_t cuptiCorrelationId_;
};

} // namespace cuda
} // namespace model
} // namespace cprof

#endif