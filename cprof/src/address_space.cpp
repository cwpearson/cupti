#include "cprof/address_space.hpp"

using json = nlohmann::json;

static std::string to_string(const AddressSpace::Type &t) {
  switch (t) {
  case AddressSpace::Type::Host:
    return "host";
  case AddressSpace::Type::CudaDevice:
    return "cuda";
  case AddressSpace::Type::CudaUVA:
    return "uva";
  case AddressSpace::Type::Unknown:
    return "unknown";
  default:
    assert(0 && "Unhandled AddressSpace::Type");
  }
}

std::string AddressSpace::str() const { return to_string(type_); }

json AddressSpace::to_json() const {
  json j;
  j["type"] = to_string(type_);
  return j;
}

std::string AddressSpace::to_json_string() const { return to_json().dump(); }

bool AddressSpace::maybe_equal(const AddressSpace &other) const {
  // assert(is_valid());
  return other == *this || is_unknown() || other.is_unknown();
}
