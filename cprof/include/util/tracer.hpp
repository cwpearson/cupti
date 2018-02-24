#ifndef UTIL_TRACER_HPP
#define UTIL_TRACER_HPP

#include <fstream>
#include <mutex>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

class Tracer {
public:
  Tracer() : enabled_(false) {}
  Tracer(const std::string &path)
      : enabled_(true), out_(std::ofstream(path)), path_(path) {}
  ~Tracer() { close(); }

  bool good() const { return out_.good(); }
  const std::string &path() const { return path_; }

  void complete_event(const std::string &name,
                      const std::vector<std::string> &categories,
                      const double timestamp, const double duration,
                      const std::string &pid, const std::string &tid) {
    using boost::property_tree::ptree;
    using boost::property_tree::write_json;

    if (!enabled_) {
      return;
    }

    std::string catStr;
    for (size_t i = 0; i < categories.size(); ++i) {
      catStr += categories[i];
      if (i + 1 < categories.size()) {
        catStr += ",";
      }
    }

    ptree pt;
    pt.put("name", name);
    pt.put("cat", catStr);
    pt.put("ph", "X");
    pt.put("ts", timestamp);
    pt.put("dur", duration);
    pt.put("pid", pid);
    pt.put("tid", tid);

    // Only one thread can write out at a time
    {
      std::lock_guard<std::mutex> guard(out_mutex_);
      write_json(out_, pt, false);
    }
  }

  void close() {
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

#endif