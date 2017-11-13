#include "allocations.hpp"
#include "env.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

const Allocations::id_type Allocations::noid = AllocationRecord::noid;

Allocations &Allocations::instance() {
  static Allocations a;
  return a;
}

Allocations::Allocations() : allocations_(map_type()) {}

std::pair<Allocations::map_type::iterator, bool>
Allocations::insert(const Allocations::value_type &v) {

  assert(v.get() && "Trying to insert invalid value");
  assert(v->pos() && "Inserting allocation at nullptr");
  if (v->size() == 0) {
    printf("WARN: inserting size %lu allocation", v->size());
  }
  const auto &valIdx = reinterpret_cast<id_type>(v.get());
  std::ofstream buf(env::output_path(), std::ofstream::app);
  buf << *v;
  buf.flush();
  std::lock_guard<std::mutex> guard(access_mutex_);
  return allocations_.insert(std::make_pair(valIdx, v));
}

std::tuple<Allocations::id_type, Allocations::value_type>
Allocations::find_live(uintptr_t pos, size_t size, const AddressSpace &as) {
  assert(pos && "No allocation at null ptr");

  std::lock_guard<std::mutex> guard(access_mutex_);

  if (allocations_.empty()) {
    return std::make_pair(noid, value_type(nullptr));
  }

  // FIXME - should be any page type
  AllocationRecord dummy(pos, size, as, Memory(Memory::Any),
                         AllocationRecord::PageType::Pageable);
  for (const auto &alloc : allocations_) {
    // printf("checkin\n");
    const auto &key = alloc.first;
    const auto &val = alloc.second;
    assert(val.get());
    if (dummy.overlaps(*val)) {
      return std::make_pair(key, val);
    }
  }
  return std::make_pair(noid, value_type(nullptr));
}

std::tuple<Allocations::id_type, Allocations::value_type>
Allocations::new_allocation(uintptr_t pos, size_t size, const AddressSpace &as,
                            const Memory &am,
                            const AllocationRecord::PageType &ty) {
  auto val = value_type(new AllocationRecord(pos, size, as, am, ty));
  assert(val.get());
  return std::make_pair(insert(val).first->first, val);
}