#ifndef VALUE_HPP
#define VALUE_HPP

#include <vector>
#include <memory>

class Value {
 public:
  uintptr_t pos_;
  size_t size_;
  size_t allocationIdx_; // allocation that this value lives in

  bool contains(const uintptr_t pos) const;
  bool overlaps(const Value &other) const;
  void depends_on(size_t id) {
    dependsOnIdx_.push_back(id);
  }
  const std::vector<size_t>& depends_on() const {
    return dependsOnIdx_;
  }
 private:
  std::vector<size_t> dependsOnIdx_; // values this value depends on
};

class Values {
 public:
  std::pair<bool, size_t> get_value(uintptr_t pos, size_t size) const;
  void push_back(const Value &v) {
    return values_.push_back(v);
  }
  size_t size() const {
    return values_.size();
  }

  Value &operator[](size_t i) {
    return values_[i];
  }
  const Value &operator[](size_t i) const {
    return values_[i];
  }

 private:
  std::vector<Value> values_;
};

#endif
