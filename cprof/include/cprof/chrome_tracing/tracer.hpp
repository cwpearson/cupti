#ifndef CPROF_CHROMETRACER_TRACER_HPP
#define CPROF_CHROMETRACER_TRACER_HPP

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace cprof {
namespace chrome_tracing {

class Tracer {
public:
  Tracer() : enabled_(false) {}
  Tracer(const std::string &path)
      : enabled_(true), out_(std::ofstream(path)), path_(path) {
    out_ << "[\n";
  }
  ~Tracer() { close(); }

  bool good() const { return out_.good(); }
  const std::string &path() const { return path_; }

  template <typename T> void write_event(const T &event) {
    if (!enabled_) {
      return;
    }
    {
      std::lock_guard<std::mutex> guard(out_mutex_);
      out_ << event.json() << ",\n";
    }
  }

  void close() {
    // out_ << "]\n"; // don't need final closing brace
    if (out_.is_open()) {
      out_.close();
    }
  }

private:
  bool enabled_;
  std::ofstream out_;
  std::string path_;
  std::mutex out_mutex_;
};

} // namespace chrome_tracing
} // namespace cprof

#endif
