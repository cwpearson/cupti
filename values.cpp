#include "values.hpp"

#include <cassert>
#include <fstream>
#include <map>

// FIXME: refactor this and other find_live and location to take a location mask
std::pair<Values::key_type, Values::value_type>
Values::find_live(uintptr_t pos, size_t size, Location loc) {
  std::lock_guard<std::mutex> guard(modify_mutex_);
  if (values_.empty())
    return std::make_pair(reinterpret_cast<Values::key_type>(nullptr),
                          std::shared_ptr<Value>(nullptr));

  Extent e(pos, size);
  for (size_t i = value_order_.size() - 1; true; i--) {
    const auto valKey = value_order_[i];
    const auto &val = values_[valKey];
    assert(val.get());
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

std::pair<Values::key_type, Values::value_type>
Values::find_live_device(const uintptr_t pos, const size_t size) {
  std::lock_guard<std::mutex> guard(modify_mutex_);
  if (values_.empty())
    return std::make_pair(reinterpret_cast<Values::key_type>(nullptr),
                          std::shared_ptr<Value>(nullptr));

  Extent e(pos, size);
  for (size_t i = value_order_.size() - 1; true; i--) {
    const auto valKey = value_order_[i];
    const auto &val = values_[valKey];
    if (val->overlaps(e) && val->location().is_device_accessible())
      return std::make_pair(valKey, val);

    if (i == 0)
      break;
  }
  return std::make_pair(reinterpret_cast<Values::key_type>(nullptr),
                        std::shared_ptr<Value>(nullptr));
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
  assert(v.get() && "Inserting invalid value!");
  const auto &valIdx = reinterpret_cast<key_type>(v.get());

  std::lock_guard<std::mutex> guard(modify_mutex_);
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