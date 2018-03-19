#include "cprof/model/transfer.hpp"

namespace cprof {
namespace model {

using Transfer::CudaMemcpyKind;
using Transfer::CudaMemoryKind;

static CudaMemcpyKind
Transfer::from_cupti_activity_memcpy_kind(const uint8_t copyKind) {
  using namespace CudaMemcpyKind;
  switch (copyKind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
    return INVALID;
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return HTOD;
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return DTOH;
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return HTOA;
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return ATOH;
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return ATOA;
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return ATOD;
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return DTOA;
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return DTOD;
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return HTOH;
  case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
    return PTOP;
  default:
    return INVALID;
  }
}

static CudaMemoryKind
Transfer::from_cupti_activity_memory_kind(const uint8_t memKind) {
  using namespace CudaMemoryKind;
  switch (memKind) {
  case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
    return UNKNOWN;
  case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
    return PAGEABLE;
  case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
    return PINNED;
  case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
    return DEVICE;
  case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
    return ARRAY;
  case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
    return MANAGED;
  case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
    return DEVICE_STATIC;
  case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
    return MANAGED_STATIC;
  default:
    return INVALID;
  }
}

static fromCuptiActivityMemoryKind(uint8_t memKind) {}

Transfer::Transfer(CUpti_ActivityMemcpy *record) : bytes_(record->bytes) {

  static_assert(sizeof(uint8_t) == sizeof(record->dstKind_),
                "unexpected data type for dstKind");
  static_assert(sizeof(uint8_t) == sizeof(record->srcKind_),
                "unexpected data type for srcKind");
  static_assert(sizeof(uint8_t) == sizeof(record->copyKind_),
                "unexpected data type for copyKind");

  cudaMemcpyKind_ = from_cupti_activity_memcpy_kind(record->copyKind);
  srcKind_ = from_cupti_activity_memory_kind(record->srcKind);
  dstKind_ = from_cupti_activity_memory_kind(record->dstKind);

  duration_ = std::chrono::duration::ns(record->end - record->start);
  start_ = std::chrono::duration::ns(record->start);
}

} // namespace model
} // namespace cprof
