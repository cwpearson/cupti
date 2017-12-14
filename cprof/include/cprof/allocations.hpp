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
#include "cprof/allocation.hpp"
#include "cprof/model/location.hpp"
#include "util/extent.hpp"
#include "util/interval_set.hpp"
#include "util/logging.hpp"

namespace cprof {
class Allocations {

public:
private:
  typedef uintptr_t pos_type;
  typedef Allocation value_type;
  typedef IntervalSet<AllocationRecord> icl_type;

private:
  std::map<AddressSpace, icl_type> addrSpaceAllocs_;
  std::mutex access_mutex_;

  Allocation insert(const AllocationRecord &ar);

  Allocation unsafe_find(uintptr_t pos, size_t size, const AddressSpace &as);

public:
  size_t size() {
    size_t tot = 0;
    for (const auto &m : addrSpaceAllocs_) {
      tot += m.second.size();
    }
    return tot;
  }

  Allocation free(uintptr_t pos, const AddressSpace &as);

  /*! \brief Lookup allocation that contains pos, size, and address space.
   */
  Allocation find(uintptr_t pos, size_t size, const AddressSpace &as) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    return unsafe_find(pos, size, as);
  }
  /*! \brief Lookup allocation containing pos and address space
   */
  Allocation find(uintptr_t pos, const AddressSpace &as) {
    return find(pos, 1, as);
  }

  Allocation new_allocation(uintptr_t pos, size_t size, const AddressSpace &as,
                            const cprof::model::Memory &am,
                            const cprof::model::Location &al);

  Value new_value(const uintptr_t pos, const size_t size,
                  const AddressSpace &as, const bool initialized);

  Value find_value(const uintptr_t pos, const size_t size,
                   const AddressSpace &as);
  Value find_value(const uintptr_t pos, const AddressSpace &as) {
    return find_value(pos, 1, as);
  }

  Value duplicate_value(const Value &v, const bool initialized) {
    // ensure the existing value exists
    auto orig = find_value(v.pos(), v.size(), v.address_space());
    assert(orig);
    return new_value(v.pos(), v.size(), v.address_space(), initialized);
  }

  Allocations() {}
  ~Allocations() { logging::err() << "DEBU: Allocations dtor\n"; }
};

} // namespace cprof
#endif
