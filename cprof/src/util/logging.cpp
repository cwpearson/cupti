#include "util/logging.hpp"

namespace logging {

Logger &globalLogger() {
  static Logger l;
  return l;
}

std::ostream &out() { return globalLogger().out(); }
std::ostream &err() { return globalLogger().err(); }
std::ostream &debug() { return globalLogger().err() << "DEBUG: "; }
std::ostream &info() { return globalLogger().err() << "INFO : "; }
std::ostream &warn() { return globalLogger().err() << "WARN : "; }
std::ostream &error() { return globalLogger().err() << "ERR  : "; }

std::ostream &set_out_path(const std::string &path) {
  return globalLogger().set_out_path(path);
}
std::ostream &set_err_path(const std::string &path) {
  return globalLogger().set_err_path(path);
}

void disable() { globalLogger().disable(); }
void enable() { globalLogger().enable(); }

void atomic_out(const std::string &s) { globalLogger().atomic_out(s); }
void atomic_err(const std::string &s) { globalLogger().atomic_err(s); }
} // namespace logging