#include "value.hpp"

bool Value::contains(const uintptr_t pos) const {
  if (pos >= pos_ && pos < pos_ + size_) {
    return true;
  } else {
    return false;
  }
}

bool Value::overlaps(const Value &other) const {      
  if (other.contains(pos_))               return true;
  if (other.contains(pos_ + size_ - 1))       return true;
  if (contains(other.pos_))               return true;
  if (contains(other.pos_ + other.size_ - 1)) return true;
  return false;
}

std::pair<bool, size_t> Values::get_value(uintptr_t pos, size_t size) const {
  if (values_.empty()) return std::make_pair(false, -1);

  Value dummy(pos, size);
  for (size_t i = values_.size() - 1; ; i--) {
    if (dummy.overlaps(values_[i])) {
      return std::make_pair(true, i);
    }  
 
    if (i == 0) break;
  }
  return std::make_pair(false, -1);
}
