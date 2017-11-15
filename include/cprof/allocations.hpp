#ifndef CPROF_ALLOCATIONS_HPP
#define CPROF_ALLOCATIONS_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "address_space.hpp"
#include "allocation_record.hpp"
#include "extent.hpp"

class Allocations {
public:
  typedef std::shared_ptr<AllocationRecord> value_type;

private:
  typedef std::vector<value_type> container_type;

public:
  typedef container_type::iterator iterator;
  typedef container_type::reverse_iterator reverse_iterator;
  typedef container_type::const_iterator const_iterator;

private:
  container_type allocations_;
  std::mutex access_mutex_;

public:
  // simple implementations
  iterator end() { return allocations_.end(); }

  /*! \brief Lookup allocation by position, size, and address space.
   */
  value_type find(uintptr_t pos, size_t size, const AddressSpace &as);
  value_type find(uintptr_t pos, const AddressSpace &as) {
    return find(pos, 0, as);
  }
  value_type find_exact(uintptr_t pos, const AddressSpace &as);

  value_type new_allocation(uintptr_t pos, size_t size, const AddressSpace &as,
                            const Memory &am,
                            const AllocationRecord::PageType &ty);

  size_t free(uintptr_t pos, const AddressSpace &as) {
    auto i = find_exact(pos, as);
    if (i) {
      i->mark_free();
      return 1;
    }
    assert(0 && "Expecting to erase an allocation.");
    return 0;
  }

  static Allocations &instance();

private:
  Allocations() {}
};

#endif
