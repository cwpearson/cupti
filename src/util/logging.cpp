#include "util/logging.hpp"

namespace logging {

Logger &globalLogger() {
  static Logger l;
  return l;
}

std::ostream &out() { return globalLogger().out(); }
std::ostream &err() { return globalLogger().err(); }
std::ostream &set_out_path(const std::string &path) {
  return globalLogger().set_out_path(path);
}
std::ostream &set_err_path(const std::string &path) {
  return globalLogger().set_err_path(path);
}
} // namespace logging