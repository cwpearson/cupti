#include "value.hpp"

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <sstream>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

void Value::add_depends_on(size_t id) {
  dependsOnIdx_.push_back(id);

  ptree pt;
  pt.put("dep.dst_id", std::to_string(Id()));
  pt.put("dep.src_id", std::to_string(id));
  std::ofstream buf(output_path, std::ofstream::app);
  write_json(buf, pt, false);
  buf.flush();
}

std::string Value::json() const {
  ptree pt;
  pt.put("val.id", std::to_string(Id()));
  pt.put("val.pos", std::to_string(pos_));
  pt.put("val.size", std::to_string(size_));
  pt.put("val.allocation_id", std::to_string(allocation_id_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}

void Value::set_size(size_t size) {
  size_ = size;
  std::ofstream buf(output_path, std::ofstream::app);
  buf << *this;
  buf.flush();
}

Value &Value::UnknownValue() {
  static Value unknown(0 /*pos*/, 0 /*size*/,
                       Allocation::UnknownAllocation().Id());
  unknown.is_unknown_ = true;
  return unknown;
}

Value &Value::NoValue() {
  static Value v(0 /*pos*/, 0 /*size*/, Allocation::NoAllocation().Id());
  v.is_not_value_ = true;
  return v;
}

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << v.json();
  return os;
}

Values::value_type Values::find_live(uintptr_t pos, size_t size, Location loc) {
  if (values_.empty())
    return std::shared_ptr<Value>(nullptr);

  Extent e(pos, size);
  for (size_t i = value_order_.size() - 1; true; i--) {
    const auto valIdx = value_order_[i];
    const auto &val = values_[valIdx];
    if (val->overlaps(e) && loc == val->location())
      return val;

    if (i == 0)
      break;
  }

  return std::shared_ptr<Value>(nullptr);
}

Values::value_type Values::find_live(uintptr_t pos, Location loc) {
  return find_live(pos, 1, loc);
}

std::pair<bool, Values::key_type>
Values::get_last_overlapping_value(uintptr_t pos, size_t size, Location loc) {
  value_type live = find_live(pos, size, loc);
  if (live.get() == nullptr) {
    return std::make_pair(false, -1);
  }

  return std::make_pair(true, live->Id());
}

std::pair<Values::map_type::iterator, bool>
Values::insert(const value_type &v) {
  const auto &valIdx = reinterpret_cast<key_type>(v.get());
  value_order_.push_back(valIdx);

  std::ofstream buf(output_path, std::ofstream::app);
  buf << *v;
  buf.flush();

  return values_.insert(std::make_pair(valIdx, v));
}

Values &Values::instance() {
  static Values v;
  return v;
}

Values::Values() : values_(map_type()), value_order_(std::vector<key_type>()) {}