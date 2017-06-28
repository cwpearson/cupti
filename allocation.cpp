#include "allocation.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

const std::string output_path("cprof.txt");

std::string to_string(const Location &l) {
  if (Location::Host == l)
    return std::string("host");
  if (Location::Device == l)
    return std::string("device");
  return std::string("<unexpected location>");
}

std::string Allocation::json() const {
  ptree pt;
  pt.put("allocation.id", std::to_string(uintptr_t(this)));
  pt.put("allocation.pos", std::to_string(pos_));
  pt.put("allocation.size", std::to_string(size_));
  pt.put("allocation.loc", to_string(location_));
  pt.put("allocation.device_id", std::to_string(deviceId_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

Allocation &Allocation::UnknownAllocation() {
  static Allocation unknown(0 /*pos*/, 0 /*size*/, Location::Host);
  unknown.is_unknown_ = true;
  return unknown;
}

Allocation &Allocation::NoAllocation() {
  static Allocation a(0 /*pos*/, 0 /*size*/, Location::Host);
  a.is_not_allocation_ = true;
  return a;
}

std::ostream &operator<<(std::ostream &os, const Allocation &v) {
  os << v.json();
  return os;
}
