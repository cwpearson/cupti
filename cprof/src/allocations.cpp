#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <cassert>

#include "cprof/allocations.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

namespace cprof {

Allocation Allocations::unsafe_find(uintptr_t pos, size_t size,
                                    const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  auto allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    auto &allocations = allocationsIter->second;

    // std::cerr << "(unsafe_find) looking for {" << as.str() << "}[" << pos
    //           << ", +" << size << ")\n";
    // std::cerr << si.lower() << " " << si.upper() << "\n";
    // for (auto &e : allocations) {
    //   std::cerr << e.first.lower() << " " << e.first.upper() << "\n";
    // }
    auto ai = allocations.find(Allocation(pos, size));
    if (ai != allocations.end()) {
      // std::cerr << "matching allocation at " << ai->second.pos() << "\n";
      return *ai;
    } else {
      // std::cerr << "no matching alloc\n";
      return Allocation();
    }
  } else {
    // std::cerr << "no matching AS\n";
    return Allocation();
  }
}

Allocation Allocations::free(uintptr_t pos, const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  auto allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    auto &allocations = allocationsIter->second;
    auto ai = allocations.find(pos);
    if (ai != allocations.end()) {
      if (ai->pos() == pos) {
        ai->free();
        return *ai;
      }
    }
  }
  return Allocation();
}

Allocation Allocations::insert(const Allocation &a) {
  logging::atomic_out(a.json());
  auto &allocs = addrSpaceAllocs_[a.address_space()];
  allocs.insert_join(a);
  return a;
}

Allocation Allocations::new_allocation(uintptr_t pos, size_t size,
                                       const AddressSpace &as, const Memory &am,
                                       const Location &al) {
  if (size == 0) {
    logging::err() << "WARN: creating size 0 allocation" << std::endl;
  }
  std::lock_guard<std::mutex> guard(access_mutex_);
  return insert(Allocation(pos, size, as, am, al));
}

Value Allocations::find_value(const uintptr_t pos, const size_t size,
                              const AddressSpace &as) {
  std::lock_guard<std::mutex> guard(access_mutex_);
  logging::err() << "INFO: Looking for value @ {" << as.str() << "}[" << pos
                 << ", +" << size << ")" << std::endl;

  const auto &alloc = unsafe_find(pos, size, as);
  if (!alloc) {
    logging::err() << "DEBU: Didn't find alloc while looking for value @ ["
                   << pos << ", +" << size << ")" << std::endl;
    return Value();
  } else {
    logging::err() << "DEBU: Found alloc [" << alloc.pos() << ", +"
                   << alloc.size() << ") while looking for value @ [" << pos
                   << ", +" << size << ")" << std::endl;
  }
  const auto val = alloc.value(pos, size);
  return val;
}

// Value Allocations::new_value(const uintptr_t pos, const size_t size,
//                              const AddressSpace &as, const bool initialized)
//                              {
//   std::lock_guard<std::mutex> guard(access_mutex_);

//   // Find the allocation
//   auto alloc = unsafe_find(pos, size, as);
//   assert(alloc && "Allocation should be valid");

//   return alloc.new_value(pos, size, initialized);
// }

} // namespace cprof
