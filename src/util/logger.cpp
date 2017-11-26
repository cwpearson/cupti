#include "util/logger.hpp"

#include <cassert>
#include <fstream>
#include <iostream>

std::ostream &Logger::out() {
  if (out_) {
    return *out_;
  } else {
    return std::cout;
  }
}
std::ostream &Logger::err() {
  if (err_)
    return *err_;
  else
    return std::cerr;
}

std::ostream &Logger::set_err_path(const std::string &path) {
  err_ = std::unique_ptr<std::ofstream>(new std::ofstream(path.c_str()));
  assert(err_->good() && "Unable to open err file");
  return err();
}

std::ostream &Logger::set_out_path(const std::string &path) {
  out_ = std::unique_ptr<std::ofstream>(new std::ofstream(path.c_str()));
  assert(out_->good() && "Unable to open out file");
  std::cerr << this << ": out_ set\n";
  return out();
}