#ifndef VALUE_HPP
#define VALUE_HPP

#include <vector>

class Value {
 public:
  uintptr_t pos_;
  size_t size_;
  size_t allocationIdx_; // allocation that this value lives in
  std::vector<size_t> dependsOnIdx_; // values this value depends on
};

typedef std::vector<Value> Values;


#endif
