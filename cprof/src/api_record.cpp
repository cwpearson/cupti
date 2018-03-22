#include <cassert>

#include <nlohmann/json.hpp>

#include "cprof/api_record.hpp"

using json = nlohmann::json;
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

void ApiRecord::set_wall_start(const cprof::time_point_t &start) {
  wallStart_ = start;
}

void ApiRecord::set_wall_end(const cprof::time_point_t &end) { wallEnd_ = end; }

void ApiRecord::set_wall_time(const cprof::time_point_t &start,
                              const cprof::time_point_t &end) {
  set_wall_start(start);
  set_wall_end(end);
}

std::string ApiRecord::to_json_string() const {

  using cprof::nanos;

  std::vector<Value::id_t> inputIds(inputs_.size());
  std::vector<Value::id_t> outputIds(outputs_.size());

  for (size_t i = 0; i < inputs_.size(); ++i) {
    inputIds[i] = inputs_[i].id();
  }

  for (size_t i = 0; i < outputs_.size(); ++i) {
    outputIds[i] = outputs_[i].id();
  }


  json j;
  j["api"]["id"] = id();
  j["api"]["name"] = apiName_;
  j["api"]["device"] = device_;
  j["api"]["symbolname"] = kernelName_;
  j["api"]["inputs"] = json(inputIds);
  j["api"]["outputs"] = json(outputIds);
  j["api"]["wall_start"] = nanos(wallStart_);
  j["api"]["wall_end"] = nanos(wallEnd_);
  j["api"]["correlation_id"] = correlationId_;
  j["api"]["kv"] = json(kv_);
  return j.dump();
}