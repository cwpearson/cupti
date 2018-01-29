#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <mutex>
#include <sstream>

#include "cprof/allocations.hpp"
#include "cprof/value.hpp"

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

std::ostream &operator<<(std::ostream &os, const cprof::Value &dt) {
  os << dt.id();
  return os;
}

namespace cprof {

void Value::add_depends_on(const Value &v, const uint64_t apiId) const {
  ptree pt;
  pt.put("dep.dst_id", v.id());
  pt.put("dep.src_id", id());
  pt.put("dep.tid", cprof::model::get_thread_id());
  pt.put("pt.api_cause", apiId);
  std::stringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

std::string Value::json() const {
  ptree pt;
  pt.put("val.id", id_);
  pt.put("val.pos", pos_);
  pt.put("val.size", size_);
  pt.put("val.allocation", allocation_);
  pt.put("val.address_space", addressSpace_.json());
  pt.put("val.initialized", initialized_);
  std::stringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

const AddressSpace &Value::address_space() const noexcept {
  return addressSpace_;
}

void Value::set_size(const size_t size) {
  logging::err() << "WARN: not updating size in Value" << std::endl;
  logging::atomic_out(json());
}

} // namespace cprof
