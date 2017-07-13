#include "allocation_record.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using boost::property_tree::ptree;
using boost::property_tree::write_json;

const std::string output_path("cprof.txt");

const AllocationRecord::id_type AllocationRecord::noid =
    reinterpret_cast<id_type>(nullptr);

AllocationRecord::AllocationRecord(uintptr_t pos, size_t size,
                                   const AddressSpace &as, const Memory &mem,
                                   PageType pt)
    : Extent(pos, size), address_space_(as), memory_(mem), type_(pt) {
  assert(address_space_.is_valid());
}

std::string to_string(AllocationRecord::PageType type) {
  if (AllocationRecord::PageType::Pinned == type)
    return "pinned";
  if (AllocationRecord::PageType::Pageable == type) {
    return "pageable";
  }
  if (AllocationRecord::PageType::Unknown == type) {
    return "unknown";
  }
  assert(0 && "Unexpected AllocationRecord::type");
}

std::string AllocationRecord::json() const {
  ptree pt;
  pt.put("allocation.id", std::to_string(uintptr_t(this)));
  pt.put("allocation.pos", std::to_string(pos_));
  pt.put("allocation.size", std::to_string(size_));
  pt.put("allocation.addrsp", address_space_.json());
  pt.put("allocation.mem", memory_.json());
  pt.put("allocation.type", to_string(type_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const AllocationRecord &v) {
  os << v.json();
  return os;
}
