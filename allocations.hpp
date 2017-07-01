#ifndef ALLOCATIONS_HPP
#define ALLOCATIONS_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>

#include "allocation.hpp"
#include "extent.hpp"
#include "location.hpp"

class Allocations {
public:
  typedef Allocation::id_type key_type;
  typedef std::shared_ptr<Allocation> value_type;

private:
  typedef std::map<key_type, value_type> map_type;
  map_type allocations_;

public:
  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::tuple<bool, key_type> find_live(uintptr_t pos, size_t size,
                                       Location location);
  std::tuple<bool, key_type> find_live(uintptr_t pos, Location loc) {
    return find_live(pos, 1, loc);
  }

  value_type &operator[](const key_type &k) { return allocations_[k]; }
  value_type &operator[](key_type &&k) { return allocations_[k]; }

  static Allocations &instance();

private:
  Allocations();
};

#endif
