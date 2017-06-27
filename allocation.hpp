#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>

#include "extent.hpp"

class Allocation : public Extent {
public:
  enum class Location { Host, Device };
  enum class Type { Pinned, Pageable };

private:
  typedef uintptr_t id_type;
  Location location_;

public:
  friend std::ostream &operator<<(std::ostream &os, const Allocation &v);
  size_t deviceId_;
  Allocation(uintptr_t pos, size_t size, Location loc)
      : Extent(pos, size), location_(loc) {}

  id_type Id() const { return reinterpret_cast<id_type>(this); }
  std::string json() const;

  bool overlaps(const Allocation &other) {
    return (location_ == other.location_) && Extent::overlaps(other);
  }

  bool contains(const Allocation &other) {
    return (location_ == other.location_) && Extent::contains(other);
  }
};

class Allocations {
private:
  typedef uintptr_t key_type;
  typedef std::shared_ptr<Allocation> value_type;
  typedef std::map<key_type, value_type> map_type;
  map_type allocations_;

public:
  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::tuple<bool, key_type> find_live(uintptr_t pos, size_t size,
                                       Allocation::Location location);

  static Allocations &instance();

private:
  Allocations();
};

#endif
