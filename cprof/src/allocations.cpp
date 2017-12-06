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

Allocation Allocations::npos_ = Allocation();

const Allocation &Allocations::find(uintptr_t pos, size_t size,
                                    const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  // std::cerr << " looking for " << pos << "\n";
  std::lock_guard<std::mutex> guard(access_mutex_);
  auto allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    auto &allocations = allocationsIter->second;

    auto si = interval<uintptr_t>::right_open(pos, pos + size);
    auto ai = allocations.find(si);
    if (ai != allocations.end()) {
      // std::cerr << "matching allocation at " << ai->second.pos() << "\n";
      return ai->second;
    } else {
      // std::cerr << "no matching alloc\n";
      return end();
    }
  } else {
    // std::cerr << "no matching AS\n";
    return end();
  }
}

const Allocation &Allocations::free(uintptr_t pos, const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  auto allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    auto &allocations = allocationsIter->second;

    auto ai = allocations.find(pos);
    if (ai != allocations.end()) {
      if (ai->second.pos() == pos) {
        const auto &alloc = ai->second;
        const auto size = alloc.size();
        const auto am = alloc.memory();
        const auto al = alloc.location();
        allocations.erase(ai->second.interval());
        return new_allocation(pos, size, as, am, al);
      } else {
        return end();
      }
    } else {
      return end();
    }
  } else {
    return end();
  }
}

const Allocation &Allocations::new_allocation(uintptr_t pos, size_t size,
                                              const AddressSpace &as,
                                              const Memory &am,
                                              const Location &al) {
  Allocation val(pos, size, as, am, al);

  if (val.size() == 0) {
    logging::err() << "WARN: creating size 0 allocation" << std::endl;
  }

  logging::atomic_out(val.json());

  std::lock_guard<std::mutex> guard(access_mutex_);
  auto &allocs = addrSpaceAllocs_[as];
  const auto &i = interval<uintptr_t>::right_open(pos, pos + size);
  allocs += std::make_pair(i, val);

  return allocs.find(i)->second;
}

} // namespace cprof