#ifndef CPROF_MODEL_LOCATION_HPP
#define CPROF_MODEL_LOCATION_HPP

namespace cprof {
namespace model {

class Location {
private:
  enum class Type {
    Unknown,
    Host,
    CudaDevice,
  };

  using Type::CudaDevice;
  using Type::Host;
  using Type::Unknown;

  Type type_;
  int id_;

  Location(const Type &type, const int id) : type_(type), id_(id) {}
  Location(const Type &type) : Location(type, -1) {}

public:
  static Location Unknown() { return Location(Unknown); }
  static Location Host() { return Location(Host); }
  static Location CudaDevice(int id) { return Location(CudaDevice, id); }
};

} // namespace model
} // namespace cprof

#endif