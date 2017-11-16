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

  /*! \brief Lookup allocation that contains pos, size.
   */
  value_type find(uintptr_t pos, size_t size);

  /*! \brief Lookup allocation that contains pos, size, and address space.
   */
  value_type find(uintptr_t pos, size_t size, const AddressSpace &as);
  /*! \brief Lookup allocation containing pos and address space
   */
  value_type find(uintptr_t pos, const AddressSpace &as) {
    return find(pos, 1, as);
  }
  /*! \brief Lookup allocation starting at pos in address space
   */
  value_type find_exact(uintptr_t pos, const AddressSpace &as);

  value_type new_allocation(uintptr_t pos, size_t size, const AddressSpace &as,
                            const cprof::model::Memory &am);

  size_t free(uintptr_t pos, const AddressSpace &as);

  static Allocations &instance();

private:
  Allocations() {}
};

#endif
