#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocations.hpp"
#include "cprof/profiler.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

Allocations &Allocations::instance() {
  static Allocations a;
  return a;
}

Allocations::value_type Allocations::find(uintptr_t pos, size_t size) {
  assert(pos && "No allocations at null pointer");

  std::vector<Allocations::value_type> matches;

  std::lock_guard<std::mutex> guard(access_mutex_);
  for (reverse_iterator i = allocations_.rbegin(), e = allocations_.rend();
       i != e; ++i) {
    Allocation a = *i;
    if (a->contains(pos, size) && !a->freed()) {
      matches.push_back(*i);
    }
  }

  if (matches.size() == 1) {
    return matches[0];
  } else if (matches.empty()) {
    return nullptr;
  } else {
    for (const auto &a : matches) {
      printf("INFO: matching %lu, %lu\n", a->pos(), a->size());
    }
    assert(0 && "Multiple matches in different address spaces!");
  }
}

Allocations::value_type Allocations::find(uintptr_t pos, size_t size,
                                          const AddressSpace &as) {
  assert(pos && "No allocations at null pointer");
  std::lock_guard<std::mutex> guard(access_mutex_);
  for (reverse_iterator i = allocations_.rbegin(), e = allocations_.rend();
       i != e; ++i) {
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
  for (reverse_iterator i = allocations_.rbegin(), e = allocations_.rend();
       i != e; ++i) {
    if ((*i)->pos() == pos && (*i)->address_space() == as && !(*i)->freed()) {
      return *i;
    }
  }
  return nullptr;
}

Allocations::value_type Allocations::new_allocation(uintptr_t pos, size_t size,
                                                    const AddressSpace &as,
                                                    const Memory &am,
                                                    const Location &al) {
  auto val = value_type(new AllocationRecord(pos, size, as, am, al));
  assert(val.get());

  if (val->size() == 0) {
    printf("WARN: creating size 0 allocation");
  }

  cprof::out() << *val;
  cprof::out().flush();

  std::lock_guard<std::mutex> guard(access_mutex_);
  allocations_.push_back(val);
  return val;
}

size_t Allocations::free(uintptr_t pos, const AddressSpace &as) {
  auto i = find_exact(pos, as);
  if (i->freed()) {
    printf("WARN: allocation %lu double-free?\n", i->pos());
  }
  if (i) {
    i->mark_free();
    return 1;
  }
  assert(0 && "Expecting to erase an allocation.");
  return 0;
}