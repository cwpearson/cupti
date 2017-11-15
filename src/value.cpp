#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <sstream>

#include "cprof/allocations.hpp"
#include "cprof/env.hpp"
#include "cprof/value.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

const Value::id_type Value::noid = reinterpret_cast<Value::id_type>(nullptr);

void Value::add_depends_on(id_type id) {
  dependsOnIdx_.push_back(id);
  ptree pt;
  pt.put("dep.dst_id", Id());
  pt.put("dep.src_id", id);
  std::ofstream buf(env::output_path(), std::ofstream::app);
  write_json(buf, pt, false);
  buf.flush();
}

std::string Value::json() const {
  ptree pt;
  pt.put("val.id", Id());
  pt.put("val.pos", pos_);
  pt.put("val.size", size_);
  pt.put("val.allocation_", allocation_);
  pt.put("val.initialized", is_initialized_);
  std::stringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

void Value::record_meta_append(const std::string &s) {
  ptree pt;
  pt.put("meta.append", s);
  pt.put("meta.val_id", Id());
  std::ofstream buf(env::output_path(), std::ofstream::app);
  write_json(buf, pt, false);
  buf.flush();
}

void Value::record_meta_set(const std::string &s) {
  ptree pt;
  pt.put("meta.set", s);
  pt.put("meta.val_id", Id());
  std::ofstream buf(env::output_path(), std::ofstream::app);
  write_json(buf, pt, false);
  buf.flush();
}

AddressSpace Value::address_space() const {
  // FIXME - allocations may be modified during this function call. Not
  // thread-safe
  assert(allocation_);
  return allocation_->address_space();
}

void Value::set_size(size_t size) {
  size_ = size;
  std::ofstream buf(env::output_path(), std::ofstream::app);
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
