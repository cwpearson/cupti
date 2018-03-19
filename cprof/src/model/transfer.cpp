#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/model/transfer.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;

namespace cprof {
namespace model {

std::string to_string(const Transfer::Kind &kind) {
  switch (kind) {
  case Transfer::Kind::CUPTI_MEMCPY:
    return "cupti_memcpy";
  case Transfer::Kind::INVALID:
    return "invalid";
  default:
    assert(0 && "unxpected Transfer::Kind");
  }
}

Transfer::Transfer() : kind_(Transfer::Kind::INVALID) {}

Transfer::Transfer(CUpti_ActivityMemcpy *record) : Transfer() {

  static_assert(sizeof(uint8_t) == sizeof(record->dstKind),
                "unexpected data type for dstKind");
  static_assert(sizeof(uint8_t) == sizeof(record->srcKind),
                "unexpected data type for srcKind");
  static_assert(sizeof(uint8_t) == sizeof(record->copyKind),
                "unexpected data type for copyKind");

  assert(record && "invalid record");

  bytes_ = record->bytes;
  cudaDeviceId_ = record->deviceId;
  kind_ = Kind::CUPTI_MEMCPY;

  cudaMemcpyKind_ = from_cupti_activity_memcpy_kind(record->copyKind);
  srcKind_ = from_cupti_activity_memory_kind(record->srcKind);
  dstKind_ = from_cupti_activity_memory_kind(record->dstKind);

  duration_ = std::chrono::nanoseconds(record->end - record->start);
  start_ = time_point_t(std::chrono::nanoseconds(record->start));
}

double Transfer::start_ns() const {
  auto startNs = std::chrono::time_point_cast<std::chrono::nanoseconds>(start_);
  auto startEpoch = startNs.time_since_epoch();
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(startEpoch);
  return value.count();
}

double Transfer::dur_ns() const {
  auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_);
  return value.count();
}

std::string Transfer::json() const {

  ptree pt;
  pt.put("transfer.cuda_device_id", cudaDeviceId_);
  pt.put("transfer.kind", to_string(kind_));
  pt.put("transfer.cuda_memcpy_kind", to_string(cudaMemcpyKind_));
  pt.put("transfer.src_kind", to_string(srcKind_));
  pt.put("transfer.dst_kind", to_string(dstKind_));
  pt.put("transfer.start", start_ns());
  pt.put("transfer.dur", dur_ns());
  for (const auto &p : kv_) {
    const std::string &key = p.first;
    const std::string &val = p.second;
    pt.put("transfer." + key, val);
  }
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

} // namespace model
} // namespace cprof
