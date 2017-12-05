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

Allocation &Allocations::find(uintptr_t pos, size_t size) {
  assert(pos && "No allocations at null pointer");

  std::vector<Allocations::value_type> matches;
  std::lock_guard<std::mutex> guard(access_mutex_);

  for (const auto &kv : addrSpaceAllocs_) {
    const auto &as = kv.first;
    const auto &allocs = kv.second;

    const auto &si = interval<uintptr_t>::right_open(pos, pos + size);
    const auto &ai = allocs.find(si);
    if (ai != allocs.end()) {
      matches.push_back(ai->second);
    }
  }

  if (matches.size() == 1) {
    return matches[0];
  } else if (matches.empty()) {
    return end();
  } else { // FIXME for now, return most recent. Issue 11. Should be fused in
           // allocation creation?
    assert(0 && "Found allocation in multiple address spaces");
  }
}

Allocation &Allocations::find(uintptr_t pos, size_t size, AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  auto allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    auto &allocations = allocationsIter->second;

    auto si = interval<uintptr_t>::right_open(pos, pos + size);
    auto ai = allocations.find(si);
    if (ai != allocations.end()) {
      return ai->second;
    } else {
      return end();
    }
  } else {
    return end();
  }
}

Allocation &Allocations::find_exact(uintptr_t pos, const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  const auto &allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    const auto &allocations = allocationsIter->second;

    const auto &ai = allocations.find(pos);
    if (ai != allocations.end()) {
      if (ai->second.pos() == pos) {
        return ai->second;
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

Allocation &Allocations::new_allocation(uintptr_t pos, size_t size,
                                        const AddressSpace &as,
                                        const Memory &am, const Location &al) {
  Allocation val(pos, size, as, am, al);

  if (val.size() == 0) {
    logging::err() << "WARN: creating size 0 allocation" << std::endl;
  }

  logging::atomic_out(val.json());

  {
    std::lock_guard<std::mutex> guard(access_mutex_);
    auto &allocs = addrSpaceAllocs_[as];
    const auto &i = interval<uintptr_t>::right_open(pos, pos + size);
    allocs += std::make_pair(i, val);
  }
  return val;
}

size_t Allocations::free(uintptr_t pos, const AddressSpace &as) {
  auto i = find_exact(pos, as);
  if (i.address_space() == as && i.freed()) {
    logging::err() << "WARN: allocation @" << i.pos() << " double-free?"
                   << std::endl;
  }
  if (i) {
    i.mark_free();
    return 1;
  }
  assert(0 && "Expecting to erase an allocation.");
  return 0;
}

} // namespace cprof