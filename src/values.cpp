#include <cassert>
#include <fstream>
#include <map>

#include "cprof/profiler.hpp"
#include "cprof/values.hpp"

// FIXME: address space
Value
Values::find_live(uintptr_t pos, size_t size, const AddressSpace &as) {
  std::lock_guard<std::mutex> guard(access_mutex_);

    auto i = interval<uintptr_t>::right_open(pos, pos + size);
    auto found = values_.find(i);
    if (found != values_.end()) {
      return found->second;
    }

    return Value(nullptr);

}

// Values::id_type Values::find_live(const uintptr_t pos, const AddressSpace
// &as,
//                                   const Memory &mem) const {}

/*
std::pair<Values::map_type::iterator, bool>
Values::insert(const value_type &v) {
  assert(v.get() && "Inserting invalid value!");
  const auto &valIdx = reinterpret_cast<id_type>(v.get());

  std::lock_guard<std::mutex> guard(access_mutex_);
  value_order_.push_back(valIdx);

  cprof::out() << *v;
  cprof::out().flush();

  return values_.insert(std::make_pair(valIdx, v));
}
*/ 

Values &Values::instance() {
  static Values v;
  return v;
}

// Values::Values() : values_(map_type()), value_order_(std::vector<id_type>()) {}