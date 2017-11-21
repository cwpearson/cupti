#ifndef BACKTRACE_HPP
#define BACKTRACE_HPP

#include <cstdint>
#include <cstdlib>
#include <execinfo.h>
#include <ostream>

void print_backtrace(std::ostream &os) {
  void *buf[256];

  const size_t sz = backtrace(buf, 256);
  auto strs = backtrace_symbols(buf, sz);

  for (size_t i = 0; i < sz; ++i) {
    os << strs[i] << std::endl;
  }

  free(strs);
}

#endif