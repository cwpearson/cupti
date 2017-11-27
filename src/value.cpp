#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <mutex>
#include <sstream>

#include "cprof/allocations.hpp"
#include "cprof/profiler.hpp"
#include "cprof/value.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

void ValueRecord::add_depends_on(const ValueRecord &vr) {
  ptree pt;
  pt.put("dep.dst_id", uintptr_t(&vr));
  pt.put("dep.src_id", uintptr_t(this));
  pt.put("dep.tid", cprof::model::get_thread_id());
  std::stringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

std::string ValueRecord::json() const {
  ptree pt;
  pt.put("val.id", uintptr_t(this));
  pt.put("val.pos", pos_);
  pt.put("val.size", size_);
  pt.put("val.allocation_", std::to_string(uintptr_t(allocation_.get())));
  pt.put("val.initialized", initialized_);
  std::stringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

/*
void VB::record_meta_append(const std::string &s) {
  ptree pt;
  pt.put("meta.append", s);
  pt.put("meta.val_id", Id());
  write_json(cprof::out(), pt, false);
  cprof::out().flush();
}

void VB::record_meta_set(const std::string &s) {
  ptree pt;
  pt.put("meta.set", s);
  pt.put("meta.val_id", Id());
  write_json(cprof::out(), pt, false);
  cprof::out().flush();
}
*/

AddressSpace ValueRecord::address_space() const {
  // FIXME - not thread-safe
  assert(allocation_);
  return allocation_->address_space();
}

void ValueRecord::set_size(const size_t size) {
  size_ = size;
  logging::atomic_out(json());
}

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << uintptr_t(v.get());
  return os;
}