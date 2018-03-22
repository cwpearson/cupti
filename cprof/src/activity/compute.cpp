#include <nlohmann/json.hpp>

#include "cprof/activity/compute.hpp"

namespace cprof {
namespace activity {

std::string to_string(const Compute::Kind &kind) {
  switch (kind) {
  case Compute::Kind::CUPTI_KERNEL3:
    return "cupti_kernel3";
  case Compute::Kind::INVALID:
    return "invalid";
  default:
    assert(0 && "unxpected Compute::Kind");
  }
}

Compute::Compute() : kind_(Compute::Kind::INVALID) {}

Compute::Compute(const CUpti_ActivityKernel3 *record) : Compute() {
  assert(record && "invalid record");

  kind_ = Kind::CUPTI_KERNEL3;
  start_ = time_point_t(std::chrono::nanoseconds(record->start));
  duration_ = std::chrono::nanoseconds(record->end - record->start);

  if (record->completed == CUPTI_TIMESTAMP_UNKNOWN) {
    completed_ = time_point_t(std::chrono::nanoseconds(0));
  } else {
    completed_ = time_point_t(std::chrono::nanoseconds(record->completed));
  }
  cudaDeviceId_ = record->deviceId;
  contextId_ = record->contextId;
  correlationId_ = record->correlationId;
  streamId_ = record->streamId;
  name_ = record->name;
}

double Compute::start_ns() const {
  auto startNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(start_);
  auto startEpoch = startNs.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(startEpoch);
  return value.count();
}

double Compute::dur_ns() const {
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_);
  return value.count();
}

double Compute::completed_ns() const {
  auto cNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(completed_);
  auto cEpoch = cNs.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(cEpoch);
  return value.count();
}

std::string Compute::to_json_string() const {
  using json = nlohmann::json;
  json j;
  j["compute"]["cuda_device_id"] = cudaDeviceId_;
  j["compute"]["kind"] = to_string(kind_);
  j["compute"]["name"] = name_;
  j["compute"]["start"] = start_ns();
  j["compute"]["dur"] = dur_ns();
  j["compute"]["completed"] = completed_ns();
  j["compute"]["stream_id"] = streamId_;
  j["compute"]["correlation_id"] = correlationId_;
  j["compute"]["kv"] = json(kv_);
  return j.dump();
}

cprof::chrome_tracing::CompleteEvent Compute::chrome_complete_event() const {

  if (completed_ns() >= start_ns()) {
    return cprof::chrome_tracing::CompleteEventNs(name_, {}, start_ns(),
                                                  completed_ns() - start_ns(),
                                                  "profiler", "compute");
  } else {
    return cprof::chrome_tracing::CompleteEventNs(
        name_, {}, start_ns(), dur_ns(), "profiler", "compute");
  }
}

} // namespace activity
} // namespace cprof
