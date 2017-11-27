#ifndef UTIL_LOGGER_HPP
#define UTIL_LOGGER_HPP

#include <memory>
#include <mutex>
#include <ostream>
#include <string>

class Logger {
private:
  std::unique_ptr<std::ostream> err_;
  std::unique_ptr<std::ostream> out_;
  std::mutex outMutex_;

public:
  Logger() : err_(nullptr), out_(nullptr) {}
  std::ostream &err();
  std::ostream &out();
  std::ostream &set_err_path(const std::string &path);
  std::ostream &set_out_path(const std::string &path);
  void atomic_out(const std::string &s);
};

#endif