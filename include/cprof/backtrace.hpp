#ifndef BACKTRACE_HPP
#define BACKTRACE_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>

void print_backtrace() {
  void *buf[256];

  const size_t sz = backtrace(buf, 256);
  auto strs = backtrace_symbols(buf, sz);

  for (size_t i = 0; i < sz; ++i) {
    printf("%s\n", strs[i]);
  }

  free(strs);
}

#endif