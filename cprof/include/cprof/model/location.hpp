#ifndef CPROF_MODEL_LOCATION_HPP
#define CPROF_MODEL_LOCATION_HPP

#include <string>

#include <nlohmann/json.hpp>

namespace cprof {
namespace model {

class Location {
  using json = nlohmann::json;

private:
  enum class Type {
    Unknown,
    Host,
    CudaDevice,
  };

  Type type_;
  int id_;

  Location(const Type &type, const int id) : type_(type), id_(id) {}
  Location(const Type &type) : Location(type, -1) {}

public:
  static Location Unknown() { return Location(Type::Unknown); }
  static Location Host(const int id) { return Location(Type::Host, id); }
  static Location CudaDevice(const int id) {
    return Location(Type::CudaDevice, id);
  }

  static std::string to_string(const Type &type) {
    switch (type) {
    case Type::Unknown:
      return "unknown";
    case Type::Host:
      return "host";
    case Type::CudaDevice:
      return "cuda";
    default:
      assert(0 && "how did we get here");
    }
  }

  json to_json() const;
  std::string to_json_string() const;
};

} // namespace model
} // namespace cprof

#endif