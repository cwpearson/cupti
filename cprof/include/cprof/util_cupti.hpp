#ifndef UTIL_CUPTI_HPP
#define UTIL_CUPTI_HPP

#include <cstdlib>
#include <ostream>
#include <string>

#include <cupti.h>

#define CUPTI_CHECK(ans, err)                                                  \
  { cuptiAssert((ans), __FILE__, __LINE__, (err)); }
inline void cuptiAssert(CUptiResult code, const char *file, int line,
                        std::ostream &err, bool abort = true) {
  if (code != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(code, &errstr);
    err << "CUPTI_CHECK: " << errstr << " " << file << " " << line << std::endl;
    if (abort)
      exit(code);
  }
}

namespace cprof {

enum class CuptiActivityMemcpyKind {
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

enum class CuptiActivityMemoryKind {
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

std::string to_string(const CuptiActivityMemcpyKind &kind);
std::string to_string(const CuptiActivityMemoryKind &kind);

// should be passed uint8_t from CUpti_Activity
// for example, Cupti_ActivityMemcpy.copyKind
CuptiActivityMemcpyKind from_cupti_activity_memcpy_kind(const uint8_t copyKind);
CuptiActivityMemoryKind from_cupti_activity_memory_kind(const uint8_t memKind);

} // namespace cprof

#endif