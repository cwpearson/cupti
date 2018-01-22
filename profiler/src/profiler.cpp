#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include "cprof/model/thread.hpp"
#include "util/environment_variable.hpp"

#include "profiler.hpp"

namespace profiler {
cprof::model::Driver &driver() { return Profiler::instance().driver_; }
cprof::model::Hardware &hardware() { return Profiler::instance().hardware_; }
cprof::Allocations &allocations() { return Profiler::instance().allocations_; }
KernelCallTime &kernelCallTime() {
  return Profiler::instance().kernelCallTime_;
}

std::ostream &out() { return Profiler::instance().out(); }
void atomic_out(const std::string &s) { Profiler::instance().atomic_out(s); }
std::ostream &err() { return Profiler::instance().err(); }
} // namespace profiler

/*! \brief Profiler() create a profiler
 *
 * Should not handle any initialization. Defer that to the init() method.
 */
Profiler::Profiler() : manager_(nullptr), isInitialized_(false) {}

Profiler::~Profiler() {
  logging::err() << "Profiler dtor\n";
  delete manager_;

  if (enableZipkin_) {
    profiler::err() << "INFO: Profiler finalizing Zipkin" << std::endl;
    memcpyTracer_->Close(); // FIXME: right order?
    launchTracer_->Close();
    rootSpan_->Finish();
    rootTracer_->Close();
  }

  isInitialized_ = false;
  logging::err() << "Profiler dtor almost done...\n";
}

std::ostream &Profiler::err() { return logging::err(); }
std::ostream &Profiler::out() { return logging::out(); }
void Profiler::atomic_out(const std::string &s) {
  return logging::atomic_out(s);
}

/*! \brief Profiler() initialize a profiler object
 *
 * Handle initialization here so that calls to Profiler member functions are
 * valid, since they've already been constructed.
 */
void Profiler::init() {
  // std::cerr << model::get_thread_id() << std::endl;

  if (isInitialized_) {
    logging::err() << "Profiler alread initialized" << std::endl;
    return;
  }

  // Configure logging
  auto outPath = EnvironmentVariable<std::string>("CPROF_OUT", "-").get();
  if (outPath != "-") {
    logging::set_out_path(outPath.c_str());
  }
  auto errPath = EnvironmentVariable<std::string>("CPROF_ERR", "-").get();
  if (errPath != "-") {
    logging::set_err_path(errPath.c_str());
  }

  err() << "INFO: Profiler::init()" << std::endl;
  const bool useCuptiCallback =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_CALLBACK", true).get();
  err() << "INFO: useCuptiCallback: " << useCuptiCallback << std::endl;

  const bool useCuptiActivity =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_ACTIVITY", true).get();
  err() << "INFO: useCuptiActivity: " << useCuptiActivity << std::endl;

  enableZipkin_ = EnvironmentVariable<bool>("CPROF_ENABLE_ZIPKIN", false).get();
  err() << "INFO: enableZipkin: " << enableZipkin_ << std::endl;

  zipkinHost_ =
      EnvironmentVariable<std::string>("CPROF_ZIPKIN_HOST", "localhost").get();
  err() << "INFO: zipkinEndpoint: " << zipkinHost_ << std::endl;

  zipkinPort_ = EnvironmentVariable<uint32_t>("CPROF_ZIPKIN_PORT", 9411u).get();
  err() << "INFO: zipkinPort: " << zipkinPort_ << std::endl;

  err() << "INFO: scanning devices" << std::endl;
  hardware_.get_device_properties();
  err() << "INFO: done" << std::endl;

  manager_ =
      new CuptiSubscriber(useCuptiActivity, useCuptiCallback, enableZipkin_);
  manager_->init();

  if (enableZipkin_) {
    profiler::err() << "INFO: Profiler enable zipkin" << std::endl;
    // Create tracers here so that they are not destroyed
    // when clearing buffer during destruction
    zipkin::ZipkinOtTracerOptions options;
    options.service_name = "profiler";
    options.collector_host = Profiler::instance().zipkin_host();
    options.collector_port = Profiler::instance().zipkin_port();
    rootTracer_ = makeZipkinOtTracer(options);

    zipkin::ZipkinOtTracerOptions memcpyTracerOpts;
    memcpyTracerOpts.service_name = "memcpy tracer";
    memcpyTracerOpts.collector_host = Profiler::instance().zipkin_host();
    memcpyTracerOpts.collector_port = Profiler::instance().zipkin_port();
    memcpyTracer_ = makeZipkinOtTracer(memcpyTracerOpts);

    zipkin::ZipkinOtTracerOptions launchTracerOpts;
    launchTracerOpts.service_name = "kernel tracer";
    launchTracerOpts.collector_host = Profiler::instance().zipkin_host();
    launchTracerOpts.collector_port = Profiler::instance().zipkin_port();
    launchTracer_ = makeZipkinOtTracer(launchTracerOpts);

    rootSpan_ = rootTracer_->StartSpan("Root");
  }

  isInitialized_ = true;
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}

static ProfilerInitializer pi;