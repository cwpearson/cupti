#ifndef UTIL_LOGGING_HPP
#define UTIL_LOGGING_HPP

#include "util/logger.hpp"

namespace logging {
std::ostream &out();
std::ostream &err();
std::ostream &set_err_path(const std::string &path);
std::ostream &set_out_path(const std::string &path);
std::ostream &debug();
void atomic_out(const std::string &s);
} // namespace logging

#endif