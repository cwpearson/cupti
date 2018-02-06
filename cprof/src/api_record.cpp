#include <cassert>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/api_record.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;
using cprof::Value;

std::atomic<ApiRecord::id_type> ApiRecord::next_id_(0);

void ApiRecord::add_input(const Value &v) {
  assert(v);
  inputs_.push_back(v);
}

void ApiRecord::add_output(const Value &v) {
  assert(v);
  outputs_.push_back(v);
}

void ApiRecord::add_kv(const std::string &k, const std::string &v) {
  kv_[k] = v;
}
void ApiRecord::add_kv(const std::string &k, const size_t &v) {
  add_kv(k, std::to_string(v));
}

static ptree to_json(const std::vector<Value> &v) {
  ptree array;
  for (const auto &e : v) {
    ptree elem;
    elem.put("", e.id());
    array.push_back(std::make_pair("", elem));
  }
  return array;
}

std::string ApiRecord::json() const {
  ptree pt;
  pt.put("api.id", id());
  pt.put("api.name", apiName_);
  pt.put("api.device", device_);
  pt.put("api.symbolname", kernelName_);
  pt.add_child("api.inputs", to_json(inputs_));
  pt.add_child("api.outputs", to_json(outputs_));
  pt.put("api.start", nanos(start_));
  pt.put("api.end", nanos(end_));
  for (const auto &p : kv_) {
    const std::string &key = p.first;
    const std::string &val = p.second;
    pt.put("api." + key, val);
  }
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}