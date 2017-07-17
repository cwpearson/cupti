
#include "output_path.hpp"

#include <cstdlib>
#include <cstring>
#include <string>

namespace output_path {

const std::string override_("");
const std::string default_("output.cprof");

const char *get() {

  // Override, if it exists
  if (!override_.empty()) {
    return override_.c_str();
  }

  // Environment variable, if it is set
  const char *fromEnv = std::getenv("CPROF_OUT");
  if (fromEnv && (0 < std::strlen(fromEnv))) {
    return fromEnv;
  }

  // default otherwise
  return default_.c_str();
}
}