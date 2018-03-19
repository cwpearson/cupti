#include <sstream>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "cprof/chrome_tracing/complete_event.hpp"

using cprof::chrome_tracing::CompleteEvent;

namespace cprof {
namespace chrome_tracing {

CompleteEvent::CompleteEvent(const std::string &name,
                             const std::vector<std::string> &categories,
                             const std::string &pid, const std::string &tid)
    : name_(name), categories_(categories), pid_(pid), tid_(tid), timestamp_(0),
      duration_(0) {}

void CompleteEvent::ts_from_us(const double timestamp) {
  timestamp_ = timestamp;
}
void CompleteEvent::dur_from_us(const double duration) { duration_ = duration; }
void CompleteEvent::ts_from_ns(const double timestamp) {
  ts_from_us(timestamp / 1e3);
}
void CompleteEvent::dur_from_ns(const double duration) {
  dur_from_us(duration / 1e3);
}

std::string CompleteEvent::json() const {
  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  std::string catStr;
  for (size_t i = 0; i < categories_.size(); ++i) {
    catStr += categories_[i];
    if (i + 1 < categories_.size()) {
      catStr += ",";
    }
  }

  ptree pt;
  pt.put("name", name_);
  pt.put("cat", catStr);
  pt.put("ph", "X");
  pt.put("ts", timestamp_);
  pt.put("dur", duration_);
  pt.put("pid", pid_);
  pt.put("tid", tid_);

  std::stringstream str;
  write_json(str, pt, false);
  return str.str();
}

CompleteEvent CompleteEventNs(const std::string &name,
                              const std::vector<std::string> &categories,
                              const double timestamp, const double duration,
                              const std::string &pid, const std::string &tid) {
  CompleteEvent event(name, categories, pid, tid);
  event.ts_from_ns(timestamp);
  event.dur_from_ns(duration);
  return event;
}

CompleteEvent CompleteEventUs(const std::string &name,
                              const std::vector<std::string> &categories,
                              const double timestamp, const double duration,
                              const std::string &pid, const std::string &tid) {
  CompleteEvent event(name, categories, pid, tid);
  event.ts_from_us(timestamp);
  event.dur_from_us(duration);
  return event;
}

} // namespace chrome_tracing
} // namespace cprof
