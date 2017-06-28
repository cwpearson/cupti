#include "values.hpp"

#include <fstream>
#include <map>

std::pair<Values::key_type, Values::value_type>
Values::find_live(uintptr_t pos, size_t size, Location loc) {
  if (values_.empty())
    return std::make_pair(reinterpret_cast<Values::key_type>(nullptr),
                          std::shared_ptr<Value>(nullptr));

  Extent e(pos, size);
  for (size_t i = value_order_.size() - 1; true; i--) {
    const auto valKey = value_order_[i];
    const auto &val = values_[valKey];
    if (val->overlaps(e) && loc == val->location())
      return std::make_pair(valKey, val);

    if (i == 0)
      break;
  }

  return std::make_pair(reinterpret_cast<Values::key_type>(nullptr),
                        std::shared_ptr<Value>(nullptr));
}

std::pair<Values::key_type, Values::value_type>
Values::find_live(uintptr_t pos, Location loc) {
  return find_live(pos, 1, loc);
}

std::pair<bool, Values::key_type>
Values::get_last_overlapping_value(uintptr_t pos, size_t size, Location loc) {
  auto kv = find_live(pos, size, loc);
  if (kv.first == uintptr_t(nullptr)) {
    return std::make_pair(false, -1);
  }

  return std::make_pair(true, kv.first);
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