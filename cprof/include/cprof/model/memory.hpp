#ifndef CPROF_MODEL_MEMORY_HPP
#define CPROF_MODEL_MEMORY_HPP

#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

namespace cprof {
namespace model {

enum class Memory {
  Unknown,    ///< an unknown type of memory
  Pageable,   ///< CUDA pageable memory
  Pagelocked, ///< CUDA Page-locked memory
  Unified3,   ///< CUDA unified memory >sm_30
  Unified6    ///< CUDA unified memory >sm_60
};

nlohmann::json to_json(const Memory &m);

} // namespace model
} // namespace cprof

#endif