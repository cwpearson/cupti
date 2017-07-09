#ifndef ALLOCATIONS_HPP
#define ALLOCATIONS_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>

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
  std::mutex access_mutex_;

public:
  // void lock() { access_mutex_.lock(); }
  // void unlock() { access_mutex_.unlock(); }

  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::tuple<bool, key_type> find_live(uintptr_t pos, size_t size,
                                       Location location);
  std::tuple<bool, key_type> find_live(uintptr_t pos, Location loc) {
    return find_live(pos, 1, loc);
  }

  size_t free(const key_type &k) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    const size_t numErased = allocations_.erase(k);
    assert(numErased);
    return numErased;
  }

  value_type &at(const key_type &k) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    return allocations_.at(k);
  }
  value_type &at(key_type &&k) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    return allocations_.at(k);
  }

  static Allocations &instance();

private:
  Allocations();
};

#endif
