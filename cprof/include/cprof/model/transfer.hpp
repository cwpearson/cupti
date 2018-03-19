#ifndef CPROF_MODEL_TRANSFER_HPP
#define CPROF_MODEL_TRANSFER_HPP

#include <cassert>
#include <chrono>
#include <map>
#include <string>

#include <cupti.h>

#include "cprof/util_cupti.hpp"

namespace cprof {
namespace model {

class Transfer {
public:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  typedef std::chrono::nanoseconds duration_t;

  enum class Kind { CUPTI_MEMCPY, INVALID };

  Transfer();
  Transfer(CUpti_ActivityMemcpy *record);

  double start_ms() const;
  double start_ns() const;
  double dur_ms() const;
  double dur_ns() const;

  std::string json() const;

private:
  size_t bytes_;
  uint32_t cudaDeviceId_;
  Kind kind_;
  cprof::CuptiActivityMemcpyKind cudaMemcpyKind_;
  cprof::CuptiActivityMemoryKind srcKind_;
  cprof::CuptiActivityMemoryKind dstKind_;

  duration_t duration_;
  time_point_t start_;

  std::map<std::string, std::string> kv_;

  //// unused CUPTI_ActivityMemcpy fields
  // uint32_t contextId_;
  // uint32_t correlationId_;
  // uint8_t flags_;
  // CUpti_ActivityKind kind_;
  // void *reserved0_;
  // uint32_t runtimeCorrelationId_;
  // uint32_t streamId_;
};

} // namespace model
} // namespace cprof

#endif