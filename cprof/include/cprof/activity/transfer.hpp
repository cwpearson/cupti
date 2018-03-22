#ifndef CPROF_ACTIVITY_TRANSFER_HPP
#define CPROF_ACTIVITY_TRANSFER_HPP

#include <cassert>
#include <chrono>
#include <map>
#include <string>

#include <cupti.h>

#include "cprof/chrome_tracing/complete_event.hpp"
#include "cprof/util_cupti.hpp"

namespace cprof {
namespace activity {

class Transfer {
public:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  typedef std::chrono::nanoseconds duration_t;

  enum class Kind { CUPTI_MEMCPY, INVALID };

  Transfer();
  explicit Transfer(const CUpti_ActivityMemcpy *record);

  double start_ns() const;
  double dur_ns() const;

  std::string to_json_string() const;
  cprof::chrome_tracing::CompleteEvent chrome_complete_event() const;

private:
  // General fields
  size_t bytes_;
  duration_t duration_;
  time_point_t start_;
  std::map<std::string, std::string> kv_;

  // CUDA-specific fields (FIXME: move to derived class)
  uint32_t cudaDeviceId_;
  Kind kind_;
  cprof::CuptiActivityMemcpyKind cudaMemcpyKind_;
  cprof::CuptiActivityMemoryKind srcKind_;
  cprof::CuptiActivityMemoryKind dstKind_;
  uint32_t contextId_;
  uint32_t correlationId_;
  uint8_t flags_;
  uint32_t runtimeCorrelationId_;
  uint32_t streamId_;
};

} // namespace activity
} // namespace cprof

#endif