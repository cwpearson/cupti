#include "allocations.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

const std::string output_path("cprof.txt");

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

  std::ofstream buf(output_path, std::ofstream::app);
  buf << *v;
  buf.flush();
  std::lock_guard<std::mutex> guard(access_mutex_);
  return allocations_.insert(std::make_pair(valIdx, v));
}

std::tuple<bool, Allocations::id_type>
Allocations::find_live(uintptr_t pos, size_t size, const AddressSpace &as) {
  assert(pos && "No allocation at null ptr");

  printf("Findin an alloc\n");
  std::lock_guard<std::mutex> guard(access_mutex_);
  printf("Got lock\n");

  if (allocations_.empty()) {
    printf("no allocs\n");
    return std::make_pair(false, -1);
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
      printf("found it\n");
      return std::make_pair(true, key);
    }
  }
  return std::make_pair(false, -1);
}