#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/model/memory.hpp"

std::string cprof::model::to_string(const Memory &m) {
  switch (m) {
  case Memory::Unknown:
    return "unknown";
  case Memory::Pageable:
    return "pageable";
  case Memory::Pagelocked:
    return "pagelocked";
  case Memory::Unified3:
    return "unified3";
  case Memory::Unified6:
    return "unified6";
  default:
    assert(0 && "Unexpected memory type");
  }
}