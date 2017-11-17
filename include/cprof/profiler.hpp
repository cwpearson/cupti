#ifndef CPROF_PROFILER_HPP
#define CPROF_PROFILER_HPP

#include "cprof/cupti_subscriber.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/hardware.hpp"

namespace cprof {
class Profiler {
public:
  ~Profiler();

  /* \brief Initialize the profiler
   *
   */
  static void init();
  static Profiler &instance();

  const std::string &output_path() { return jsonOutputPath_; }

  friend model::Hardware &hardware();
  friend model::Driver &driver();

private:
  Profiler();
  model::Hardware hardware_;
  model::Driver driver_;
  CuptiSubscriber *manager_;

  bool useCuptiCallback_;
  bool useCuptiActivity_;
  std::string jsonOutputPath_;
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