#include <cassert>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/api_record.hpp"



using boost::property_tree::ptree;
using boost::property_tree::write_json;

const ApiRecord::id_type ApiRecord::noid =
    reinterpret_cast<ApiRecord::id_type>(nullptr);

void ApiRecord::add_input(const Value &v) {
  assert(v);
  inputs_.push_back(v);
}

void ApiRecord::add_output(const Value &v) {
  assert(v);
  outputs_.push_back(v);
}

void ApiRecord::record_start_time(const uint64_t start) { start_ = start; }
void ApiRecord::record_end_time(const uint64_t end) { end_ = end; }

static ptree to_json(const std::vector<Value> &v) {
  ptree array;
  for (const auto &e : v) {
    ptree elem;
    elem.put("", e);
    array.push_back(std::make_pair("", elem));
  }
  return array;
}

std::string ApiRecord::json() const {
  ptree pt;
  pt.put("api.id", Id());
  pt.put("api.name", apiName_);
  pt.put("api.device", device_);
  pt.put("api.symbolname", kernelName_);
  pt.add_child("api.inputs", to_json(inputs_));
  pt.add_child("api.outputs", to_json(outputs_));
  pt.put("api.start", start_);
  pt.put("api.end", end_);
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

std::ostream &operator<<(std::ostream &os, const ApiRecord &r) {
  os << r.json();
  return os;
}