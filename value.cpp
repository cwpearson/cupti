#include "value.hpp"
#include "allocations.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <sstream>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

void Value::add_depends_on(id_type id) {
  dependsOnIdx_.push_back(id);

  ptree pt;
  pt.put("dep.dst_id", std::to_string(uintptr_t(this)));
  pt.put("dep.src_id", std::to_string(id));
  std::ofstream buf(output_path, std::ofstream::app);
  write_json(buf, pt, false);
  buf.flush();
}

std::string Value::json() const {
  ptree pt;
  pt.put("val.id", std::to_string(uintptr_t(this)));
  pt.put("val.pos", std::to_string(pos_));
  pt.put("val.size", std::to_string(size_));
  pt.put("val.allocation_id", std::to_string(allocation_id_));
  pt.put("val.initialized", std::to_string(is_initialized_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

Location Value::location() const {
  return Allocations::instance()[allocation_id_]->location();
}

void Value::set_size(size_t size) {
  size_ = size;
  std::ofstream buf(output_path, std::ofstream::app);
  buf << *this;
  buf.flush();
}

// Value &Value::UnknownValue() {
//   static Value unknown(0 /*pos*/, 0 /*size*/,
//                        Allocation::UnknownAllocation().Id());
//   unknown.is_unknown_ = true;
//   return unknown;
// }

// Value &Value::NoValue() {
//   static Value v(0 /*pos*/, 0 /*size*/, Allocation::NoAllocation().Id());
//   v.is_not_value_ = true;
//   return v;
// }

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << v.json();
  return os;
}
