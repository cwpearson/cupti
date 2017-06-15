#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP

#include <cstdlib>
#include <cstdint>
#include <map>
#include <memory>

class Allocation {
 public:
  int type_;
  uintptr_t pos_;
  size_t size_;
  Allocation(uintptr_t pos, size_t size) : pos_(pos), size_(size) {}

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
