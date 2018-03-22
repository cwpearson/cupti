#include <map>
#include <mutex>
#include <sstream>

#include "cprof/allocations.hpp"
#include "cprof/value.hpp"

using json = nlohmann::json;

std::ostream &operator<<(std::ostream &os, const cprof::Value &dt) {
  os << dt.id();
  return os;
}

namespace cprof {

json Value::to_json() const {
  json j;
  j["val"]["id"] = id_;
  j["val"]["pos"] = pos_;
  j["val"]["size"] = size_;
  j["val"]["allocation"] = allocation_;
  j["val"]["address_space"] = addressSpace_.to_json();
  j["val"]["initialized"] = initialized_;
  return j;
}

std::string Value::to_json_string() const { return to_json().dump(); }

const AddressSpace &Value::address_space() const noexcept {
  return addressSpace_;
}

void Value::set_size(const size_t size) {
  logging::err() << "WARN: not updating size in Value" << std::endl;
  logging::atomic_out(to_json_string() + "\n");
}

bool Value::operator==(const Value &rhs) const {
  if (id_ == rhs.id_) {
    assert(allocation_ == rhs.allocation_);
    assert(addressSpace_ == rhs.addressSpace_);
    assert(initialized_ == rhs.initialized_);
    return true;
  }
  return false;
}

bool Value::operator!=(const Value &rhs) const { return !((*this) == rhs); }

void to_json(json &j, const Value &v) { j = v.to_json(); }

} // namespace cprof
