#ifndef CPROF_CHROMETRACER_COMPLETEEVENT_HPP
#define CPROF_CHROMETRACER_COMPLETEEVENT_HPP

#include <fstream>
#include <mutex>
#include <string>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace cprof {
namespace chrome_tracing {

class CompleteEvent {
public:
  std::string name_;
  std::vector<std::string> categories_;
  std::string pid_;
  std::string tid_;

private:
  double timestamp_;
  double duration_;

public:
  /* \brief Construct a complete event with invalid times
   *
   */
  CompleteEvent(const std::string &name,
                const std::vector<std::string> &categories,
                const std::string &pid, const std::string &tid);

  /*! \brief Set timestamp from provided nanoseconds.
   */
  void ts_from_ns(const double timestamp);
  /*! \brief Set duration from provided nanoseconds.
   */
  void dur_from_ns(const double duration);
  /*! \brief Set timestamp from provided microseconds.
   */
  void ts_from_us(const double timestamp);
  /*! \brief Set duration from provided microseconds.
   */
  void dur_from_us(const double duration);

  std::string json() const;
};

/* \brief Construct a complete event with times given in ns
 *
 */
CompleteEvent CompleteEventNs(const std::string &name,
                              const std::vector<std::string> &categories,
                              const double timestamp, const double duration,
                              const std::string &pid, const std::string &tid);

/* \brief Construct a complete event with times given in ns
 *
 */
CompleteEvent CompleteEventUs(const std::string &name,
                              const std::vector<std::string> &categories,
                              const double timestamp, const double duration,
                              const std::string &pid, const std::string &tid);

} // namespace chrome_tracing
} // namespace cprof

#endif