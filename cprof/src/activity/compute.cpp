#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/activity/compute.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;

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

  completed_ = time_point_t(std::chrono::nanoseconds(record->completed));
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

std::string Compute::json() const {
  ptree pt;
  pt.put("compute.cuda_device_id", cudaDeviceId_);
  pt.put("compute.kind", to_string(kind_));
  pt.put("compute.name", name_);
  pt.put("compute.start", start_ns());
  pt.put("compute.dur", dur_ns());
  pt.put("compute.completed", completed_ns());
  pt.put("compute.stream_id", streamId_);
  pt.put("compute.correlation_id", correlationId_);
  for (const auto &p : kv_) {
    const std::string &key = p.first;
    const std::string &val = p.second;
    pt.put("compute." + key, val);
  }
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

cprof::chrome_tracing::CompleteEvent Compute::chrome_complete_event() const {
  return cprof::chrome_tracing::CompleteEventNs(name_, {}, start_ns(),
                                                completed_ns() - start_ns(),
                                                "profiler", "compute");
}

} // namespace activity
} // namespace cprof
