#include "value.hpp"

#include <sstream>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
using boost::property_tree::write_json;

bool Value::contains(const uintptr_t pos) const {
  if (pos >= pos_ && pos < pos_ + size_) {
    return true;
  } else {
    return false;
  }
}

bool Value::overlaps(const Value &other) const {      
  if (other.contains(pos_))                   return true;
  if (other.contains(pos_ + size_ - 1))       return true;
  if (contains(other.pos_))                   return true;
  if (contains(other.pos_ + other.size_ - 1)) return true;
  return false;
}

void Value::depends_on(size_t id) {
  dependsOnIdx_.push_back(id);

  ptree pt;
  pt.put("dep.src_id", std::to_string(Id()));
  pt.put("dep.dst_id", std::to_string(id));
  std::ofstream buf(output_path, std::ofstream::app); 
  write_json (buf, pt, false);
  buf.flush();
}


std::string Value::json() const {
  ptree pt;
  pt.put("val.id", std::to_string(Id()));
  pt.put("val.pos", std::to_string(pos_));
  pt.put("val.size", std::to_string(size_));
  pt.put("val.allocation_id", std::to_string(allocation_id_));
  std::ostringstream buf;
  write_json (buf, pt, false);
  return buf.str();
}


void Value::set_size(size_t size) {
  size_ = size;
  std::ofstream buf(output_path, std::ofstream::app);
  buf << *this;
  buf.flush();
}

std::ostream &operator<<(std::ostream &os, const Value &v) {
  os << v.json();
  return os;
}


std::pair<bool, Values::key_type> Values::get_last_overlapping_value(uintptr_t pos, size_t size) {
  if (values_.empty()) return std::make_pair(false, -1);

  Value dummy(pos, size);
  for (size_t i = value_order_.size() - 1; ; i--) {
    const auto valIdx = value_order_[i];
    if (dummy.overlaps(*values_[valIdx])) {
      return std::make_pair(true, valIdx);
    }  
 
    if (i == 0) break;
  }
  return std::make_pair(false, -1);
}


std::pair<Values::map_type::iterator, bool> Values::insert(const value_type &v) {
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

