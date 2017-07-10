#include "memory.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

const Memory::loc_t Memory::Unknown = 0x0;
const Memory::loc_t Memory::Host = 0x1;
const Memory::loc_t Memory::CudaDevice = 0x2;
const Memory::loc_t Memory::Any = 0xFFFFFFFF;

static std::string to_string(const optional<int> &o) {
  fprintf(stderr, "in here\n");
  if (o) {
    return std::to_string(o.value());
  } else {
    return "<no value>";
  }
  fprintf(stderr, "donen");
}

static std::string to_string(const Memory::loc_t &l) {
  std::string ret;
  if (l & Memory::Unknown) {
    ret += "unknown";
  }
  if (l & Memory::Host) {
    if (!ret.empty())
      ret += "|";
    ret += "host";
  }
  if (l & Memory::CudaDevice) {
    if (!ret.empty())
      ret += "|";
    ret += "cudadevice";
  }
  return ret;
}

std::string Memory::json() const {
  ptree pt;
  pt.put("loc", to_string(loc_));
  pt.put("id", to_string(id_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}