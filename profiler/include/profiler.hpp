#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <ostream>

#include "cprof/allocations.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/hardware.hpp"
#include "util/environment_variable.hpp"
#include "util/logging.hpp"

#include "cupti_subscriber.hpp"

namespace profiler {
cprof::model::Driver &driver();
cprof::model::Hardware &hardware();
cprof::Allocations &allocations();
KernelCallTime &kernelCallTime();

std::ostream &out();
void atomic_out(const std::string &s);
std::ostream &err();
} // namespace profiler

class Profiler {
  friend cprof::model::Driver &profiler::driver();
  friend cprof::model::Hardware &profiler::hardware();
  friend cprof::Allocations &profiler::allocations();
  friend KernelCallTime &profiler::kernelCallTime();

public:
  ~Profiler();

  /* \brief Initialize the profiler
   *
   */
  void init();
  static Profiler &instance();

  const std::string &zipkin_host() { return zipkinHost_; }
  const uint32_t &zipkin_port() { return zipkinPort_; }

  std::ostream &err();
  std::ostream &out();
  void atomic_out(const std::string &s);

  CuptiSubscriber *manager_; // FIXME: make this private and add an accessor

private:
  Profiler();

  cprof::model::Hardware hardware_;
  cprof::model::Driver driver_;
  cprof::Allocations allocations_;
  KernelCallTime kernelCallTime_;

  std::string zipkinHost_;
  uint32_t zipkinPort_;

  bool isInitialized_;
};

/* \brief Runs Profiler::init() at load time
 */
class ProfilerInitializer {
public:
  ProfilerInitializer() {
    Profiler &p = Profiler::instance();
    p.init();
  }
};

#endif