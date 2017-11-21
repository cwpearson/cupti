#ifndef CPROF_PROFILER_HPP
#define CPROF_PROFILER_HPP

#include <ostream>

#include "cprof/cupti_subscriber.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/hardware.hpp"
#include "util/environment_variable.hpp"

namespace cprof {
class Profiler {
public:
  ~Profiler();

  /* \brief Initialize the profiler
   *
   */
  void init();
  static Profiler &instance();

  const std::string &zipkin_host() { return zipkinHost_; }
  const uint32_t &zipkin_port() { return zipkinPort_; }

  std::ostream &out() {
    if (out_)
      return *out_;
    else
      return std::cout;
  }
  std::ostream &err() {
    if (err_)
      return *err_;
    else
      return std::cerr;
  }

  friend model::Hardware &hardware();
  friend model::Driver &driver();
  friend std::ostream &out();
  friend std::ostream &err();

  CuptiSubscriber *manager_; // FIXME: make this private and add an accessor

private:
  Profiler();
  model::Hardware hardware_;
  model::Driver driver_;

  bool useCuptiCallback_;
  bool useCuptiActivity_;
  std::string zipkinHost_;
  uint32_t zipkinPort_;

  std::unique_ptr<std::ostream> err_;
  std::unique_ptr<std::ostream> out_;
  bool isInitialized_;
};

inline model::Hardware &hardware() { return Profiler::instance().hardware_; }
inline model::Driver &driver() { return Profiler::instance().driver_; }
inline std::ostream &out() { return Profiler::instance().out(); }
inline std::ostream &err() { return Profiler::instance().err(); }

/* \brief Runs Profiler::init() at load time
 */
class ProfilerInitializer {
public:
  ProfilerInitializer() {
    Profiler &p = Profiler::instance();
    p.init();
  }
};

} // namespace cprof

#endif