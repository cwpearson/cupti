#include "values.hpp"
#include "env.hpp"

#include <cassert>
#include <fstream>
#include <map>

const Values::id_type Values::noid = Value::noid;

// FIXME: refactor this and other find_live and AddressSpace to take a
// AddressSpace mask
std::pair<Values::id_type, Values::value_type>
Values::find_live(uintptr_t pos, size_t size, const AddressSpace &as) {
  std::lock_guard<std::mutex> guard(access_mutex_);
  if (values_.empty())
    return std::make_pair(reinterpret_cast<Values::id_type>(nullptr),
                          std::shared_ptr<Value>(nullptr));

  Extent e(pos, size);
  for (size_t i = value_order_.size() - 1; true; i--) {
    const auto valKey = value_order_[i];
    const auto &val = values_[valKey];
    assert(val.get());
    if (val->overlaps(e) && as.maybe_equal(val->address_space()))
      return std::make_pair(valKey, val);

    if (i == 0)
      break;
  }

  return std::make_pair(reinterpret_cast<Values::id_type>(nullptr),
                        std::shared_ptr<Value>(nullptr));
}

std::pair<Values::id_type, Values::value_type>
Values::find_live(uintptr_t pos, const AddressSpace &as) {
  return find_live(pos, 1, as);
}

std::pair<Values::id_type, Values::value_type>
Values::find_live_device(const uintptr_t pos, const size_t size) {
  std::lock_guard<std::mutex> guard(access_mutex_);
  if (values_.empty())
    return std::make_pair(reinterpret_cast<Values::id_type>(nullptr),
                          std::shared_ptr<Value>(nullptr));

  Extent e(pos, size);
  for (size_t i = value_order_.size() - 1; true; i--) {
    const auto valKey = value_order_[i];
    const auto &val = values_[valKey];
    if (val->overlaps(e) && val->address_space().is_cuda())
      return std::make_pair(valKey, val);

    if (i == 0)
      break;
  }
  return std::make_pair(reinterpret_cast<Values::id_type>(nullptr),
                        std::shared_ptr<Value>(nullptr));
}

std::pair<bool, Values::id_type>
Values::get_last_overlapping_value(uintptr_t pos, size_t size,
                                   const AddressSpace &as) {
  auto kv = find_live(pos, size, as);
  if (kv.first == uintptr_t(nullptr)) {
    return std::make_pair(false, -1);
  }

  return std::make_pair(true, kv.first);
}

// Values::id_type Values::find_live(const uintptr_t pos, const AddressSpace
// &as,
//                                   const Memory &mem) const {}

std::pair<Values::map_type::iterator, bool>
Values::insert(const value_type &v) {
  assert(v.get() && "Inserting invalid value!");
  const auto &valIdx = reinterpret_cast<id_type>(v.get());

  std::lock_guard<std::mutex> guard(access_mutex_);
  value_order_.push_back(valIdx);

  std::ofstream buf(env::output_path(), std::ofstream::app);
  buf << *v;
  buf.flush();

  return values_.insert(std::make_pair(valIdx, v));
}

Values &Values::instance() {
  static Values v;
  return v;
}

Values::Values() : values_(map_type()), value_order_(std::vector<id_type>()) {}