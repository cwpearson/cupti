#ifndef CPROF_MODEL_TRANSFER_HPP
#define CPROF_MODEL_TRANSFER_HPP

#include <cassert>
#include <chrono>
#include <string>

#include <cupti.h>

namespace cprof {
namespace model {

class Transfer {
public:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  typedef std::chrono::nanoseconds duration_t;

  enum class Kind { CUPTI_MEMCPY };
  enum class CudaMemcpyKind {
    UNKNOWN,
    HTOD,
    DTOH,
    HTOA,
    ATOH,
    ATOA,
    ATOD,
    DTOA,
    DTOD,
    HTOH,
    PTOP,
    INVALID
  };
  enum class CudaMemoryKind {
    UNKNOWN,
    PAGEABLE,
    PINNED,
    DEVICE,
    ARRAY,
    MANAGED,
    DEVICE_STATIC,
    MANAGED_STATIC,
    INVALID
  };

  Transfer(CUpti_ActivityMemcpy *record);

  static CudaMemoryKind from_cupti_activity_memcpy_kind(const uint8_t copyKind);
  static CudaMemcpyKind from_cupti_activity_memory_kind(const uint8_t memKind);

  double start_ms();
  double start_ns();
  double dur_ms();
  double dur_ns();

  std::string json();

private:
  duration_t duration_;
  time_point_t start_;

  size_t bytes_;
  CudaMemcpyKind cudaMemcpyKind_;

  // unused CUPTI_ActivityMemcpy fields
  // uint32_t contextId_;
  // uint32_t correlationId_;
  // uint32_t deviceId_;
  // uint8_t flags_;
  // CUpti_ActivityKind kind_;
  // void *reserved0_;
  // uint32_t runtimeCorrelationId_;
  // uint32_t streamId_;
}

} // namespace model
} // namespace cprof

#endif