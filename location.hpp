#ifndef LOCATION_HPP
#define LOCATION_HPP

#include <string>

#include "set_device.hpp"

class Location {
public:
  enum class Location_t { Device, Host, Unified };

  Location() : device_(SetDevice().current_device()) {}
  Location(Location_t type) : Location() { type_ = type; }
  Location(const Location &other)
      : type_(other.type_), device_(other.device_) {}

  static Location Host() { return Location(Location_t::Host); }
  static Location Device() { return Location(Location_t::Device); }
  static Location Unified() { return Location(Location_t::Unified); }

  int device() const { return device_; }
  Location_t type() const { return type_; }

  bool operator==(const Location &rhs) const {
    return type_ == rhs.type_ && device_ == rhs.device_;
  }

  std::string str() const {
    if (Location_t::Host == type_)
      return std::string("host");
    if (Location_t::Device == type_)
      return std::string("device");
    if (Location_t::Unified == type_)
      return std::string("unified");
    return std::string("<unexpected location>");
  }

private:
  Location_t type_;
  int device_;
};

#endif