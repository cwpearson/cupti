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

  Type type_;
  int id_;

  Location(const Type &type, const int id) : type_(type), id_(id) {}
  Location(const Type &type) : Location(type, -1) {}

public:
  static Location Unknown() { return Location(Type::Unknown); }
  static Location Host() { return Location(Type::Host); }
  static Location CudaDevice(int id) { return Location(Type::CudaDevice, id); }

  std::string str() const {
    switch (type_) {
    case Type::Unknown:
      return "unknown";
    case Type::Host:
      return "host";
    case Type::CudaDevice:
      return "CudaDevice" + std::to_string(id_);
    default:
      assert(0 && "how did we get here");
    }
  }
};

} // namespace model
} // namespace cprof

#endif