#ifndef CPROF_PROFILER_HPP
#define CPROF_PROFILER_HPP

#include "cprof/allocations.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/hardware.hpp"
#include "cprof/cupti_subscriber.hpp"


namespace cprof {
class Profiler {
public:
  ~Profiler();

  /* \brief Initialize the profiler
   *
   */
  static void init();
  static Profiler &instance();

  friend model::Hardware &hardware();
  friend model::Driver &driver();

private:
  Profiler();
  model::Hardware hardware_;
  model::Driver driver_;
  CuptiSubscriber *manager_;
};

inline model::Hardware &hardware() { return Profiler::instance().hardware_; }
inline model::Driver &driver() { return Profiler::instance().driver_; }

/* \brief Runs Profiler::init() at load time
 */
class ProfilerInitializer {
public:
  ProfilerInitializer() { Profiler::init(); }
};

} // namespace cprof

#endif