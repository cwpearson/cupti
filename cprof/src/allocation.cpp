#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocation.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

size_t Allocation::next_val_ = 1;

std::string Allocation::json() const {
  ptree pt;
  pt.put("allocation.id", std::to_string(uintptr_t(this)));
  pt.put("allocation.pos", std::to_string(pos()));
  pt.put("allocation.size", std::to_string(size()));
  pt.put("allocation.addrsp", address_space_.json());
  pt.put("allocation.mem", cprof::model::to_string(memory()));
  pt.put("allocation.loc", location_.json());
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

uintptr_t Allocation::pos() const noexcept { return lower_; }
size_t Allocation::size() const noexcept { return upper_ - lower_; }
AddressSpace Allocation::address_space() const { return address_space_; }
Memory Allocation::memory() const { return memory_; }
Location Allocation::location() const { return location_; }
bool Allocation::freed() const noexcept { return freed_; }
void Allocation::free() { freed_ = true; }

// bool Allocation::operator==(const Allocation &rhs) const {
//   return ar_ == rhs.ar_;
// }