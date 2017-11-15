#include "cprof/extent.hpp"

#include <cassert>

bool Extent::contains(const Extent::pos_t pos) const {
  if (pos >= pos_ && pos < pos_ + size_) {
    return true;
  } else {
    return false;
  }
}

bool Extent::contains(const Extent &other) const {
  assert(size_ && "Extent of size 0?");
  assert(other.size_ && "Extent of size 0?");
  return contains(other.pos_) && contains(other.pos_ + other.size_ - 1);
}

bool Extent::overlaps(const Extent &other) const {
  if (other.contains(pos_))
    return true;
  if (other.contains(pos_ + size_ - 1))
    return true;
  if (contains(other.pos_))
    return true;
  if (contains(other.pos_ + other.size_ - 1))
    return true;
  return false;
}