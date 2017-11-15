#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocations.hpp"
#include "cprof/env.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

Allocations &Allocations::instance() {
  static Allocations a;
  return a;
}

Allocations::value_type Allocations::find(uintptr_t pos, size_t size,
                                          const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  for (iterator i = allocations_.end() - 1, b = allocations_.begin(); i != b;
       --i) {
    Allocation a = *i;
    if (a->contains(pos, size, as) && !a->freed()) {
      return *i;
    }
  }
  return nullptr;
}

Allocations::value_type Allocations::find_exact(uintptr_t pos,
                                                const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  for (iterator i = allocations_.end(), b = allocations_.begin(); i != b; --i) {
    if ((*i)->pos() == pos && (*i)->address_space() == as && !(*i)->freed()) {
      return *i;
    }
  }
  return nullptr;
}

Allocations::value_type
Allocations::new_allocation(uintptr_t pos, size_t size, const AddressSpace &as,
                            const Memory &am,
                            const AllocationRecord::PageType &ty) {
  auto val = value_type(new AllocationRecord(pos, size, as, am, ty));
  assert(val.get());

  if (val->size() == 0) {
    printf("WARN: creating size 0 allocation");
  }

  std::ofstream buf(env::output_path(), std::ofstream::app);
  buf << *val;
  buf.flush();

  std::lock_guard<std::mutex> guard(access_mutex_);
  allocations_.push_back(val);
  return val;
}
