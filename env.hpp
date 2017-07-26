#ifndef ENV_HPP
#define ENV_HPP

#include <cstdlib>
#include <cstring>
#include <string>

#define READ_ENV_STR(env_var, fname, default)                                  \
  inline std::string fname() {                                                 \
    const char *fromEnv = std::getenv(env_var);                                \
    if (fromEnv && (0 < std::strlen(fromEnv))) {                               \
      return std::string(fromEnv);                                             \
    } else {                                                                   \
      return default;                                                          \
    }                                                                          \
  }

namespace env {
READ_ENV_STR("CPROF_OUT", output_path, "")
}

#endif