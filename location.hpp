#ifndef LOCATION_HPP
#define LOCATION_HPP

#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <map>
#include <sstream>

using boost::property_tree::ptree;
using boost::property_tree::write_json;

#include "set_device.hpp"

class Location {
public:
  enum class Location_t { Device, Host, Unified };

  Location() {}
  Location(Location_t type, int device) : type_(type), device_(device) {}
  Location(const Location &other)
      : type_(other.type_), device_(other.device_) {}

  static Location Host(int device) {
    return Location(Location_t::Host, device);
  }
  static Location Device(int device) {
    return Location(Location_t::Device, device);
  }
  static Location Unified(int device) {
    return Location(Location_t::Unified, device);
  }

  int device() const { return device_; }
  Location_t type() const { return type_; }

  bool operator==(const Location &rhs) const {
    return type_ == rhs.type_ && device_ == rhs.device_;
  }

  bool is_host() const { return Location_t::Host == type_; }
  bool is_device_accessible() const {
    return Location_t::Device == type_ || Location_t::Unified == type_;
  }

  std::string json() const {
    ptree pt;
    pt.put("type", to_string(type_));
    pt.put("id", std::to_string(device_));
    std::ostringstream buf;
    write_json(buf, pt, false);
    return buf.str();
  }

private:
  std::string to_string(const Location_t &l) const {
    if (Location_t::Host == l)
      return std::string("host");
    if (Location_t::Device == l)
      return std::string("device");
    if (Location_t::Unified == l)
      return std::string("unified");
    assert(0 && "unhandled Location_t");
  }

  Location_t type_;
  int device_;
};

#endif