#ifndef CPROF_ACTIVITY_COMPUTE_HPP
#define CPROF_ACTIVITY_COMPUTE_HPP

#include <cassert>
#include <chrono>
#include <map>
#include <string>

#include <cupti.h>

#include "cprof/chrome_tracing/complete_event.hpp"
#include "cprof/util_cupti.hpp"

namespace cprof {
namespace activity {

class Compute {
public:
  typedef std::chrono::high_resolution_clock::time_point time_point_t;
  typedef std::chrono::nanoseconds duration_t;

  enum class Kind { CUPTI_KERNEL3, INVALID };

  Compute();
  explicit Compute(const CUpti_ActivityKernel3 *record);

  double start_ns() const;
  double dur_ns() const;
  double completed_ns() const;

  std::string json() const;
  cprof::chrome_tracing::CompleteEvent chrome_complete_event() const;

private:
  // General fields
  Kind kind_;
  time_point_t start_;
  duration_t duration_;
  std::map<std::string, std::string> kv_;

  // CUDA-specific fields (FIXME: move to derived class)
  uint32_t cudaDeviceId_;
  uint32_t contextId_;
  uint32_t correlationId_;
  uint32_t streamId_;
  time_point_t completed_; // uint64_t completed;
  std::string name_;

  // // unused fields
  // uint8_t requested_;
  // int32_t blockX_;
  // int32_t blockY_;
  // int32_t blockZ_;
  // int32_t dynamicSharedMemory_;
  // uint8_t executed_;
  // int64_t gridId_;
  // int32_t gridX_;
  // int32_t gridY_;
  // int32_t gridZ_;
  // uint32_t localMemoryPerThread_;
  // uint32_t localMemoryTotal_;
  // CUpti_ActivityPartitionedGlobalCacheConfig partitionedGlobalCacheExecuted_;
  // CUpti_ActivityPartitionedGlobalCacheConfig
  // partitionedGlobalCacheRequested_; uint16_t registersPerThread_; void
  // *reserved0_; uint8_t sharedMemoryConfig_; int32_t staticSharedMemory_;
};

} // namespace activity
} // namespace cprof

#endif