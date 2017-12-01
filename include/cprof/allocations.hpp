#ifndef CPROF_ALLOCATIONS_HPP
#define CPROF_ALLOCATIONS_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include <boost/icl/interval_map.hpp>

#include "address_space.hpp"
#include "allocation.hpp"
#include "cprof/model/location.hpp"
#include "util/extent.hpp"
#include "util/logging.hpp"

class Allocations {

public:
private:
  typedef uintptr_t pos_type;
  typedef Allocation value_type;
  typedef boost::icl::interval_map<pos_type, Allocation> icl_type;

private:
  std::map<AddressSpace, icl_type> addrSpaceAllocs_;
  std::mutex access_mutex_;

public:
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
                            const cprof::model::Memory &am,
                            const cprof::model::Location &al);

  size_t free(uintptr_t pos, const AddressSpace &as);

  // static Allocations &instance();

  Allocations() {}
  ~Allocations() { logging::err() << "DEBU: Allocations dtor\n"; }
};

#endif
