#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>

#include "extent.hpp"
#include "location.hpp"

class Allocation : public Extent {
public:
  enum class Type { Pinned, Pageable };
  typedef uintptr_t id_type;

private:
  bool is_unknown_;
  bool is_not_allocation_;
  Location location_;

public:
  friend std::ostream &operator<<(std::ostream &os, const Allocation &v);
  Allocation(uintptr_t pos, size_t size, Location loc)
      : Extent(pos, size), is_unknown_(false), is_not_allocation_(false),
        location_(loc) {}

  std::string json() const;

  bool overlaps(const Allocation &other) {
    return (location_ == other.location_) && Extent::overlaps(other);
  }

  bool contains(const Allocation &other) {
    return (location_ == other.location_) && Extent::contains(other);
  }

  id_type Id() const { return reinterpret_cast<id_type>(this); }
  Location location() const { return location_; }

  // static Allocation &UnknownAllocation();
  // static Allocation &NoAllocation();
};

#endif
