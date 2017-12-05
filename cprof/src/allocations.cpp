#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocations.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

namespace cprof {

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
  } else { // FIXME for now, return most recent. Issue 11
    logging::err() << "ERR: looking for [" << pos << ", + " << size << ")"
                   << std::endl;
    for (const auto &a : matches) {
      logging::err() << "ERR: matching " << a->pos() << " , " << a->size()
                     << std::endl;
    }
    return matches[matches.size() - 1];
  }
}

Allocations::value_type Allocations::find(uintptr_t pos, size_t size,
                                          const AddressSpace &as) {
  std::lock_guard<std::mutex> guard(access_mutex_);
  assert(pos && "No allocations at null pointer");
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
    const Allocation &A = *i;
    assert(A);
    if (A->pos() == pos && A->address_space() == as && !A->freed()) {
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

  if (val->size() == 0) {
    logging::err() << "WARN: creating size 0 allocation" << std::endl;
  }

  logging::atomic_out(val->json());

  {
    std::lock_guard<std::mutex> guard(access_mutex_);
    allocations_.push_back(val);
  }
  return val;
}

size_t Allocations::free(uintptr_t pos, const AddressSpace &as) {
  auto i = find_exact(pos, as);
  if (i->freed()) {
    logging::err() << "WARN: allocation " << i->pos() << " double-free?"
                   << std::endl;
  }
  if (i) {
    i->mark_free();
    return 1;
  }
  assert(0 && "Expecting to erase an allocation.");
  return 0;
}

}