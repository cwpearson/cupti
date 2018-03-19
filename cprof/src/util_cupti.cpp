#include <cassert>
#include <cstdlib>
#include <ostream>

#include "cprof/util_cupti.hpp"

namespace cprof {
std::string to_string(const CuptiActivityMemcpyKind &kind) {

  switch (kind) {
  case CuptiActivityMemcpyKind::UNKNOWN:
    return "unknown";
  case CuptiActivityMemcpyKind::HTOD:
    return "htod";
  case CuptiActivityMemcpyKind::DTOH:
    return "dtoh";
  case CuptiActivityMemcpyKind::HTOA:
    return "htoa";
  case CuptiActivityMemcpyKind::ATOH:
    return "atoh";
  case CuptiActivityMemcpyKind::ATOA:
    return "atoa";
  case CuptiActivityMemcpyKind::ATOD:
    return "atod";
  case CuptiActivityMemcpyKind::DTOA:
    return "dtoa";
  case CuptiActivityMemcpyKind::DTOD:
    return "dtod";
  case CuptiActivityMemcpyKind::HTOH:
    return "htoh";
  case CuptiActivityMemcpyKind::PTOP:
    return "ptop";
  case CuptiActivityMemcpyKind::INVALID:
    return "invalid";
  default:
    assert(0 && "unexpected CuptiActivityMemcpyKind");
  }
}

std::string to_string(const CuptiActivityMemoryKind &kind) {
  switch (kind) {
  case CuptiActivityMemoryKind::UNKNOWN:
    return "unknown";
  case CuptiActivityMemoryKind::PAGEABLE:
    return "pageable";
  case CuptiActivityMemoryKind::PINNED:
    return "pinned";
  case CuptiActivityMemoryKind::DEVICE:
    return "device";
  case CuptiActivityMemoryKind::ARRAY:
    return "array";
  case CuptiActivityMemoryKind::MANAGED:
    return "managed";
  case CuptiActivityMemoryKind::DEVICE_STATIC:
    return "device-static";
  case CuptiActivityMemoryKind::MANAGED_STATIC:
    return "managed-static";
  case CuptiActivityMemoryKind::INVALID:
    return "invalid";
  default:
    assert(0 && "unexpected CuptiActivityMemoryKind");
  }
}

CuptiActivityMemcpyKind
from_cupti_activity_memcpy_kind(const uint8_t copyKind) {
  switch (copyKind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
    return CuptiActivityMemcpyKind::INVALID;
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return CuptiActivityMemcpyKind::HTOD;
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return CuptiActivityMemcpyKind::DTOH;
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return CuptiActivityMemcpyKind::HTOA;
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return CuptiActivityMemcpyKind::ATOH;
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return CuptiActivityMemcpyKind::ATOA;
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return CuptiActivityMemcpyKind::ATOD;
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return CuptiActivityMemcpyKind::DTOA;
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return CuptiActivityMemcpyKind::DTOD;
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return CuptiActivityMemcpyKind::HTOH;
  case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
    return CuptiActivityMemcpyKind::PTOP;
  default:
    return CuptiActivityMemcpyKind::INVALID;
  }
}

CuptiActivityMemoryKind from_cupti_activity_memory_kind(const uint8_t memKind) {
  switch (memKind) {
  case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
    return CuptiActivityMemoryKind::UNKNOWN;
  case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
    return CuptiActivityMemoryKind::PAGEABLE;
  case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
    return CuptiActivityMemoryKind::PINNED;
  case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
    return CuptiActivityMemoryKind::DEVICE;
  case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
    return CuptiActivityMemoryKind::ARRAY;
  case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
    return CuptiActivityMemoryKind::MANAGED;
  // case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
  //   return CuptiActivityMemoryKind::DEVICE_STATIC;
  // case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
  //   return CuptiActivityMemoryKind::MANAGED_STATIC;
  default:
    return CuptiActivityMemoryKind::INVALID;
  }
}

} // namespace cprof
