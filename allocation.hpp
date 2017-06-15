#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP

#include <cstdlib>
#include <cstdint>
#include <map>
#include <memory>
#include <iostream>


class Allocation {
 private:
  typedef uintptr_t id_type;
 public:
  friend std::ostream &operator<<(std::ostream &os, const Allocation &v);
  enum class Location {Host, Device};
  enum class Type {Pinned, Pageable};
  Location location_;
  size_t deviceId_;
  uintptr_t pos_;
  size_t size_;
  Allocation(uintptr_t pos, size_t size) : pos_(pos), size_(size) {}

  id_type Id() const {
    return reinterpret_cast<id_type>(this);
  }
  std::string json() const;

};

class Allocations {
 private:
  typedef uintptr_t key_type;
  typedef std::shared_ptr<Allocation> value_type;
  typedef std::map<key_type, value_type> map_type;
  map_type allocations_;

 public:
  std::pair<map_type::iterator, bool> insert(const value_type &v);

  static Allocations& instance();

 private:
  Allocations();
};

#endif
