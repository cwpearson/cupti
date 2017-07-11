#ifndef ALLOCATIONS_HPP
#define ALLOCATIONS_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>

#include "address_space.hpp"
#include "allocation_record.hpp"
#include "extent.hpp"

class Allocations {
public:
  typedef AllocationRecord::id_type id_type;
  typedef std::shared_ptr<AllocationRecord> value_type;
  static const id_type noid;

private:
  typedef std::map<id_type, value_type> map_type;
  map_type allocations_;
  std::mutex access_mutex_;

public:
  // void lock() { access_mutex_.lock(); }
  // void unlock() { access_mutex_.unlock(); }

  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::tuple<id_type, value_type> find_live(uintptr_t pos, size_t size,
                                            const AddressSpace &as);
  std::tuple<id_type, value_type> find_live(uintptr_t pos,
                                            const AddressSpace &as) {
    return find_live(pos, 1, as);
  }

  std::tuple<id_type, value_type>
  new_allocation(uintptr_t pos, size_t size, const AddressSpace &as,
                 const Memory &am, const AllocationRecord::PageType &ty);

  size_t free(const id_type &k) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    const size_t numErased = allocations_.erase(k);
    assert(numErased);
    return numErased;
  }

  value_type &at(const id_type &k) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    return allocations_.at(k);
  }
  value_type &at(id_type &&k) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    return allocations_.at(k);
  }

  static Allocations &instance();

private:
  Allocations();
};

#endif
