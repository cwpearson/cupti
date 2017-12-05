#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/allocation.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;
using cprof::model::Location;
using cprof::model::Memory;

std::string Allocation::json() const {
  ptree pt;
  pt.put("allocation.id", std::to_string(uintptr_t(this)));
  pt.put("allocation.pos", std::to_string(pos()));
  pt.put("allocation.size", std::to_string(size()));
  pt.put("allocation.addrsp", address_space_.json());
  pt.put("allocation.mem", cprof::model::to_string(memory_));
  pt.put("allocation.loc", location_.json());
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}
