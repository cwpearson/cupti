#ifndef CPROF_TIME_HPP
#define CPROF_TIME_HPP

#include <chrono>

namespace cprof {
typedef std::chrono::time_point<std::chrono::high_resolution_clock>
    time_point_t;

inline uint64_t nanos(const time_point_t &t) {
  const auto n = std::chrono::time_point_cast<std::chrono::nanoseconds>(t);
  return n.time_since_epoch().count();
}

inline time_point_t now() { return std::chrono::high_resolution_clock::now(); }
} // namespace cprof
#endif