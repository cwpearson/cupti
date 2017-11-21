#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <sstream>

#include "cprof/allocations.hpp"
#include "cprof/profiler.hpp"
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
  write_json(cprof::out(), pt, false);
  cprof::out().flush();
}

std::string Value::json() const {
  ptree pt;
  pt.put("val.id", Id());
  pt.put("val.pos", pos_);
  pt.put("val.size", size_);
  pt.put("val.allocation_", std::to_string(uintptr_t(allocation_.get())));
  // pt.put("val.allocation_", "ah");
  pt.put("val.initialized", is_initialized_);
  std::stringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

void Value::record_meta_append(const std::string &s) {
  ptree pt;
  pt.put("meta.append", s);
  pt.put("meta.val_id", Id());
  write_json(cprof::out(), pt, false);
  cprof::out().flush();
}

void Value::record_meta_set(const std::string &s) {
  ptree pt;
  pt.put("meta.set", s);
  pt.put("meta.val_id", Id());
  write_json(cprof::out(), pt, false);
  cprof::out().flush();
}

AddressSpace Value::address_space() const {
  // FIXME - allocations may be modified during this function call. Not
  // thread-safe
  assert(allocation_);
  return allocation_->address_space();
}

void Value::set_size(size_t size) {
  size_ = size;
  cprof::out() << *this;
  cprof::out().flush();
}

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << v.json();
  return os;
}
