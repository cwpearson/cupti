#ifndef UTIL_CUPTI_HPP
#define UTIL_CUPTI_HPP

#include <cstdio>
#include <cstdlib>
#include <cupti.h>

#define CUPTI_CHECK(ans)                                                       \
  { cuptiAssert((ans), __FILE__, __LINE__); }
inline void cuptiAssert(CUptiResult code, const char *file, int line,
                        bool abort = true) {
  if (code != CUPTI_SUCCESS) {
    const char *errstr;
    cuptiGetResultString(code, &errstr);
    fprintf(stderr, "CUPTI_CHECK: %s %s %d\n", errstr, file, line);
    if (abort)
      exit(code);
  }
}

#endif