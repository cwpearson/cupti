#ifndef UTIL_CUPTI_HPP
#define UTIL_CUPTI_HPP

#include <cstdlib>
#include <cupti.h>
#include <ostream>

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

#endif