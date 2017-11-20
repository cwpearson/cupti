#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>

#include "cprof/model/location.hpp"

using boost::property_tree::ptree;
using boost::property_tree::write_json;
using cprof::model::Location;

std::string Location::json() const {
  ptree pt;
  pt.put("type", to_string(type_));
  pt.put("id", std::to_string(id_));
  std::ostringstream buf;
  write_json(buf, pt, false);
  return buf.str();
}