#ifndef DATA_HPP
#define DATA_HPP

#include "allocation.hpp"
#include "value.hpp"

class Data {
 public:
  Allocations allocations_;
  Values values_;

  static Data& instance();
 private:
  Data();
};

#endif
