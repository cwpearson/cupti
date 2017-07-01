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