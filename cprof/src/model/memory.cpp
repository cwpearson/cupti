#include "cprof/model/memory.hpp"

using json = nlohmann::json;

json cprof::model::to_json(const Memory &m) {
  json j;
  switch (m) {
  case Memory::Unknown:
    j["type"] = "unknown";
  case Memory::Pageable:
    j["type"] = "pageable";
  case Memory::Pagelocked:
    j["type"] = "pagelocked";
  case Memory::Unified3:
    j["type"] = "unified3";
  case Memory::Unified6:
    j["type"] = "unified6";
  default:
    assert(0 && "Unexpected memory type");
  }
  return j;
}