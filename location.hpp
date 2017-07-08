#ifndef LOCATION_HPP
#define LOCATION_HPP

#include <map>
#include <string>

#include "driver_state.hpp"

class Location {
public:
  typedef uint64_t flag_t;

  static constexpr flag_t Host = 0x1;
  static constexpr flag_t CudaDevice = 0x2;
  static constexpr flag_t CudaUnified = 0x4;

  Location() {}
  Location(flag_t type, int device) : type_(type), device_(device) {}
  Location(const Location &other)
      : type_(other.type_), device_(other.device_) {}

  int device() const { return device_; }

  bool operator==(const Location &rhs) const {
    return type_ == rhs.type_ && device_ == rhs.device_;
  }

  bool is_host() const { return type_ & Host; }
  bool is_device() const { return type_ & CudaDevice; }
  bool is_unified() const { return type_ & CudaUnified; }
  bool is_host_accessible() const { return is_host() || is_unified(); }
  bool is_device_accessible() const { return is_device() || is_unified(); }
  bool overlaps(const Location &other) const {
    return (is_host_accessible() && other.is_host_accessible()) ||
           (is_device_accessible() && other.is_device_accessible());
  }

  std::string json() const;

private:
  static std::string to_string(const flag_t &f);
  flag_t type_;
  int device_;
};

#endif
