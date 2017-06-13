#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP

#include <cstdlib>
#include <cstdint>
#include <vector>

class Allocation {
 public:
  int type_;
  uintptr_t pos_;
  size_t size_;
};

typedef std::vector<Allocation> Allocations;

#endif
