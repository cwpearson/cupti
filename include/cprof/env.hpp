/*

  

*/

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
READ_ENV_STR("CPROF_OUT", output_path, "output.cprof")
READ_ENV_STR("CPROF_ZIPKIN_ENDPOINT", zipkin_endpoint, "")
READ_ENV_STR("CPROF_MONGODB_ENDPOINT", mongodb_endpoint, "")
READ_ENV_STR("CPROF_USE_ZIPKIN", use_zipkin, "1")
READ_ENV_STR("CPROF_USE_JSON", use_json, "1")
}

#endif