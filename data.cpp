#include "data.hpp"

#include <cstdio>

Data::Data() : allocations_(Allocations()), values_(Values()) {
  fprintf(stderr, "%lu\n", allocations_.size());
}

Data& Data::instance() {
  static Data inst;
  return inst;
}
