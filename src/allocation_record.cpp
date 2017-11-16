#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocation_record.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;
using cprof::model::Memory;

AllocationRecord::AllocationRecord(uintptr_t pos, size_t size,
                                   const AddressSpace &as, const Memory &mem)
    : Extent(pos, size), address_space_(as), memory_(mem), freed_(false) {
  assert(address_space_.is_valid());
}

std::string AllocationRecord::json() const {
  ptree pt;
  pt.put("allocation.id", std::to_string(uintptr_t(this)));
  pt.put("allocation.pos", std::to_string(pos_));
  pt.put("allocation.size", std::to_string(size_));
  pt.put("allocation.addrsp", address_space_.json());
  pt.put("allocation.mem", cprof::model::to_string(memory_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const AllocationRecord &v) {
  os << v.json();
  return os;
}
