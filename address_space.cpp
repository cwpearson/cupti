#include "address_space.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>

using boost::property_tree::ptree;
using boost::property_tree::write_json;

std::string AddressSpace::to_string(const AddressSpace::flag_t &f) {
  std::string ret("");
  if (Host & f) {
    ret += std::string("host");
  }
  if (CudaDevice & f) {
    if (!ret.empty()) {
      ret += std::string("|");
    }
    ret += std::string("device");
  }
  if (CudaUnified & f) {
    if (!ret.empty()) {
      ret += std::string("|");
    }
    ret += std::string("unified");
  }
  return ret;
}

std::string AddressSpace::json() const {
  ptree pt;
  pt.put("type", to_string(type_));
  pt.put("id", std::to_string(device_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}