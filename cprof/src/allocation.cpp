#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocation.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

std::string AllocationRecord::json() const {
  ptree pt;
  pt.put("allocation.id", std::to_string(uintptr_t(this)));
  pt.put("allocation.pos", std::to_string(pos_));
  pt.put("allocation.size", std::to_string(size_));
  pt.put("allocation.addrsp", address_space_.json());
  pt.put("allocation.mem", cprof::model::to_string(memory_));
  pt.put("allocation.loc", location_.json());
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

uintptr_t Allocation::pos() const noexcept { return ar_->pos_; }
size_t Allocation::size() const noexcept { return ar_->size_; }
AddressSpace Allocation::address_space() const { return ar_->address_space_; }
Memory Allocation::memory() const { return ar_->memory_; }
Location Allocation::location() const { return ar_->location_; }
bool Allocation::freed() const noexcept { return ar_->freed_; }
void Allocation::free() { ar_->freed_ = true; }

bool Allocation::operator==(const Allocation &rhs) const {
  return ar_ == rhs.ar_;
}