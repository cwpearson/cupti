#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>

#include "cprof/address_space.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;

static std::string to_string(const AddressSpace::Type &t) {
  switch (t) {
  case AddressSpace::Type::Host:
    return "host";
  case AddressSpace::Type::CudaDevice:
    return "cuda";
  case AddressSpace::Type::CudaUVA:
    return "uva";
  case AddressSpace::Type::Unknown:
    return "unknown";
  default:
    assert(0 && "Unhandled AddressSpace::Type");
  }
}

std::string AddressSpace::str() const { return to_string(type_); }

std::string AddressSpace::json() const {
  ptree pt;
  pt.put("type", to_string(type_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

bool AddressSpace::maybe_equal(const AddressSpace &other) const {
  // assert(is_valid());
  return other == *this || is_unknown() || other.is_unknown();
}
