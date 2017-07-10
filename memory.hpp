#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "optional.hpp"
#include <cstdint>

class Memory {
public:
  typedef uint64_t loc_t;
  static constexpr loc_t Unknown = 0x0;
  static constexpr loc_t Host = 0x1;
  static constexpr loc_t CudaDevice = 0x2;
  static constexpr loc_t Any = 0xFFFFFFFF;

  /*
  numa region, device id, etc
  */
  loc_t loc_;
  optional<int> id_;

  Memory() : loc_(Unknown) {}
  Memory(const loc_t &loc) : loc_(loc) {}
  Memory(const loc_t &loc, int id) : loc_(loc), id_(id) {}
};

#endif