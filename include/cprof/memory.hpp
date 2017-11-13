#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <cstdint>
#include <string>

#include "optional.hpp"

class Memory {
public:
  typedef uint64_t loc_t;
  static const loc_t Unknown;
  static const loc_t Host;
  static const loc_t CudaDevice;
  static const loc_t CudaUnified;
  static const loc_t CudaAny;
  static const loc_t Any;

  /*
  numa region, device id, etc
  */
  loc_t loc_;
  optional<int> id_;

  Memory() : loc_(Unknown) {}
  Memory(const loc_t &loc) : loc_(loc) {}
  Memory(const loc_t &loc, int id) : loc_(loc), id_(id) {}

  std::string json() const;
};

#endif