#ifndef PROFILER_HPP
#define PROFILER_HPP

#include <atomic>
#include <memory>
#include <ostream>
#include <thread>

#include <zipkin.hpp>

#include "cprof/allocations.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/hardware.hpp"
#include "util/environment_variable.hpp"
#include "util/logging.hpp"
#include "util/tracer.hpp"

#include "cupti_activity.hpp"
#include "timer.hpp"

namespace profiler {
cprof::model::Driver &driver();
cprof::model::Hardware &hardware();
cprof::Allocations &allocations();
Timer &timer();

std::ostream &out();
std::ostream &err();
void atomic_out(const std::string &s);
} // namespace profiler

class Profiler {
  friend cprof::model::Driver &profiler::driver();
  friend cprof::model::Hardware &profiler::hardware();
  friend cprof::Allocations &profiler::allocations();
  friend Timer &profiler::timer();

public:
  ~Profiler();

  /* \brief Initialize the profiler
   *
   */
  void init();
  static Profiler &instance();

  const std::string &zipkin_host() { return zipkinHost_; }
  const uint32_t &zipkin_port() { return zipkinPort_; }
  bool is_zipkin_enabled() const { return enableZipkin_; }
  bool is_mode_timeline() const { return mode_ == Mode::ActivityTimeline; }
  bool is_mode_full() const { return mode_ == Mode::Full; }

  Tracer &chrome_tracer() { return *chromeTracer_; }

  std::ostream &err();
  std::ostream &out();
  void atomic_out(const std::string &s);

  std::shared_ptr<opentracing::Tracer> rootTracer_;
  std::shared_ptr<opentracing::Tracer> memcpyTracer_;
  std::shared_ptr<opentracing::Tracer> launchTracer_;
  std::shared_ptr<opentracing::Tracer> overheadTracer_;

  span_t rootSpan_;

private:
  Profiler();
  enum class Mode { Full, ActivityTimeline };

  // CUDA model
  cprof::model::Hardware hardware_;
  cprof::model::Driver driver_;
  cprof::Allocations allocations_;

  // from environment variables
  bool enableZipkin_;
  std::string zipkinHost_;
  uint32_t zipkinPort_;
  Mode mode_;
  std::vector<CUpti_ActivityKind> cuptiActivityKinds_;

  Timer timer_;
  std::shared_ptr<Tracer> chromeTracer_;

  bool isInitialized_;
  CUpti_SubscriberHandle cuptiCallbackSubscriber_;
};

class ProfilerInitializer {
public:
  ProfilerInitializer() {
    Profiler &p = Profiler::instance();
    p.init();
  }
};

#endif