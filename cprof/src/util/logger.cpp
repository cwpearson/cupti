#include "util/logger.hpp"

#include <cassert>
#include <fstream>
#include <iostream>

void Logger::enable_out(const bool enable) { outEnabled_ = enable; }
void Logger::enable_err(const bool enable) { errEnabled_ = enable; }

std::ostream &Logger::out() {
  if (outEnabled_) {
    if (out_) {
      return *out_;
    } else {
      return std::cout;
    }
  } else {
    return nullStream_;
  }
}
std::ostream &Logger::err() {
  if (errEnabled_) {
    if (err_)
      return *err_;
    else
      return std::cerr;
  } else {
    return nullStream_;
  }
}

std::ostream &Logger::set_err_path(const std::string &path) {
  err_ = std::unique_ptr<std::ofstream>(new std::ofstream(path.c_str()));
  return err();
}

std::ostream &Logger::set_out_path(const std::string &path) {
  out_ = std::unique_ptr<std::ofstream>(new std::ofstream(path.c_str()));
  return out();
}

void Logger::atomic_out(const std::string &s) {
  if (outEnabled_) {
    std::lock_guard<std::mutex> guard(outMutex_);
    out() << s;
  }
}

void Logger::atomic_err(const std::string &s) {
  if (errEnabled_) {
    std::lock_guard<std::mutex> guard(errMutex_);
    err() << s;
  }
}