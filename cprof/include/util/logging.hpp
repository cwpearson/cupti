#ifndef UTIL_LOGGING_HPP
#define UTIL_LOGGING_HPP

#include "util/logger.hpp"

namespace logging {
std::ostream &out();
std::ostream &err();
std::ostream &set_err_path(const std::string &path);
std::ostream &set_out_path(const std::string &path);
void disable_out();
void enable_out();
void disable_err();
void enable_err();
std::ostream &debug();
std::ostream &info();
std::ostream &warn();
std::ostream &error();
void atomic_out(const std::string &s);
void atomic_err(const std::string &s);
} // namespace logging

#endif