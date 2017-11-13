#ifndef EXTENT_HPP
#define EXTENT_HPP

#include <cstdint>

class Extent {
private:
  typedef uintptr_t pos_t;

protected:
  pos_t pos_;
  std::size_t size_;

public:
  Extent(pos_t pos, std::size_t size) : pos_(pos), size_(size) {}
  bool overlaps(const Extent &other) const;
  bool contains(const Extent &other) const;
  bool contains(pos_t pos) const;

  const pos_t &pos() const { return pos_; }
  const std::size_t &size() const { return size_; }
};

#endif