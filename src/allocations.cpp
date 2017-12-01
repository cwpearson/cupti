#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocations.hpp"
#include "cprof/profiler.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

Allocations::value_type Allocations::find(uintptr_t pos, size_t size) {
  assert(pos && "No allocations at null pointer");

  std::vector<Allocations::value_type> matches;
  std::lock_guard<std::mutex> guard(access_mutex_);

  for (const auto &kv : addrSpaceAllocs_) {
    const auto &as = kv.first;
    const auto &allocs = kv.second;

    const auto &ai = allocs.find(Allocation(pos, size));
    if (ai != allocs.end()) {
      matches.push_back(*ai);
    }
  }

  if (matches.size() == 1) {
    return matches[0];
  } else if (matches.empty()) {
    return Allocation();
  } else { // FIXME for now, return most recent. Issue 11. Should be fused in
           // allocation creation?
    assert(0 && "Found allocation in multiple address spaces");
  }
}

Allocations::value_type Allocations::find(uintptr_t pos, size_t size,
                                          const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  const auto &allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    const auto &allocations = allocationsIter->second;

    const auto &ai = allocations.find(Allocation(pos, size));
    if (ai != allocations.end()) {
      return *ai;
    } else {
      return Allocation();
    }
  } else {
    return Allocation();
  }
}

Allocations::value_type Allocations::find_exact(uintptr_t pos,
                                                const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  const auto &allocationsIter = addrSpaceAllocs_.find(as);
  if (allocationsIter != addrSpaceAllocs_.end()) {
    const auto &allocations = allocationsIter->second;

    const auto &ai = allocations.find(pos);
    if (ai != allocations.end()) {
      if (ai->pos() == pos)
        return *ai;
    } else {
      return Allocation();
    }
  } else {
    return Allocation();
  }
}

Allocations::value_type Allocations::new_allocation(uintptr_t pos, size_t size,
                                                    const AddressSpace &as,
                                                    const Memory &am,
                                                    const Location &al) {
  Allocation val(pos, size, as, am, al);

  if (val.size() == 0) {
    cprof::err() << "WARN: creating size 0 allocation" << std::endl;
  }

  logging::atomic_out(val.json());

  {
    std::lock_guard<std::mutex> guard(access_mutex_);
    addrSpaceAllocs_[as].insert(val);
  }
  return val;
}

size_t Allocations::free(uintptr_t pos, const AddressSpace &as) {
  auto i = find_exact(pos, as);
  if (i.address_space() == as && !i.freed()) {
    cprof::err() << "WARN: allocation " << i.pos() << " double-free?"
                 << std::endl;
  }
  if (i) {
    i.mark_free();
    return 1;
  }
  assert(0 && "Expecting to erase an allocation.");
  return 0;
}