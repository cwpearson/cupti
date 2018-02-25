#include <cassert>
#include <fstream>
#include <iostream>

#include "cprof/time.hpp"
#include "cprof/util_cupti.hpp"
#include "util/environment_variable.hpp"
#include "util/tracer.hpp"

#include "cupti_activity_tracing.hpp"
#include "cupti_callback.hpp"
#include "preload_cublas.hpp"
#include "preload_cudnn.hpp"
#include "preload_nccl.hpp"
#include "profiler.hpp"

namespace profiler {
cprof::model::Driver &driver() { return Profiler::instance().driver_; }
cprof::model::Hardware &hardware() { return Profiler::instance().hardware_; }
cprof::Allocations &allocations() { return Profiler::instance().allocations_; }
Timer &timer() { return Profiler::instance().timer_; }

std::ostream &out() { return Profiler::instance().out(); }
void atomic_out(const std::string &s) { Profiler::instance().atomic_out(s); }
std::ostream &err() { return Profiler::instance().err(); }
} // namespace profiler

/*! \brief Profiler() create a profiler
 *
 * Should not handle any initialization. Defer that to the init() method.
 */
Profiler::Profiler() : chromeTracer_(new Tracer()), isInitialized_(false) {}

Profiler::~Profiler() {
  logging::err() << "Profiler dtor\n";

  switch (mode_) {
  case Mode::ActivityTimeline:
  case Mode::Full:
    err() << "INFO: CuptiSubscriber cleaning up activity API";
    cuptiActivityFlushAll(0);
    err() << "INFO: done cuptiActivityFlushAll" << std::endl;
    profiler::err() << "INFO: CuptiSubscriber Deactivating callback API!"
                    << std::endl;
    CUPTI_CHECK(cuptiUnsubscribe(cuptiCallbackSubscriber_), err());
    profiler::err() << "INFO: done deactivating callbacks!" << std::endl;
    break;
  default: { assert(0 && "Unexpected mode"); }
  }

  if (enableZipkin_) {
    profiler::err() << "INFO: Profiler finalizing Zipkin" << std::endl;
    rootSpan_->Finish();
    memcpyTracer_->Close(); // FIXME: right order?
    launchTracer_->Close();
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
    logging::err() << "Profiler already initialized" << std::endl;
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
  {
    auto chromeTracingPath =
        EnvironmentVariable<std::string>("CPROF_CHROME_TRACING", "").get();
    if (chromeTracingPath != "") {
      chromeTracer_ = std::make_shared<Tracer>(chromeTracingPath.c_str());
    }
  }

  {
    auto n = EnvironmentVariable<uint32_t>("CPROF_CUPTI_DEVICE_BUFFER_SIZE", 0)
                 .get();
    if (n != 0) {
      cupti_activity_config::set_device_buffer_size(n);
    }
    err() << "INFO: CUpti activity device buffer size: "
          << *cupti_activity_config::attr_device_buffer_size() << std::endl;
  }

  enableZipkin_ = EnvironmentVariable<bool>("CPROF_ENABLE_ZIPKIN", false).get();
  err() << "INFO: enableZipkin: " << enableZipkin_ << std::endl;

  zipkinHost_ =
      EnvironmentVariable<std::string>("CPROF_ZIPKIN_HOST", "localhost").get();
  err() << "INFO: zipkinEndpoint: " << zipkinHost_ << std::endl;

  zipkinPort_ = EnvironmentVariable<uint32_t>("CPROF_ZIPKIN_PORT", 9411u).get();
  err() << "INFO: zipkinPort: " << zipkinPort_ << std::endl;

  std::string mode =
      EnvironmentVariable<std::string>("CPROF_MODE", "full").get();
  err() << "INFO: mode: " << mode << std::endl;
  if (mode == "activity_timeline") {
    mode_ = Mode::ActivityTimeline;
    cuptiActivityKinds_ = {
        CUPTI_ACTIVITY_KIND_KERNEL,          CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_DRIVER,          CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, CUPTI_ACTIVITY_KIND_OVERHEAD};
  } else if (mode == "full") {
    mode_ = Mode::Full;
    cuptiActivityKinds_ = {
        CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY,
        CUPTI_ACTIVITY_KIND_ENVIRONMENT, // not compatible on minsky2
        CUPTI_ACTIVITY_KIND_CUDA_EVENT,  // FIXME:available before cuda9?
        CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_RUNTIME,
        CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
        // CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS, // causes a hang in nccl on
        // minsky2
        CUPTI_ACTIVITY_KIND_OVERHEAD};
  } else {
    assert(0 && "Unsupported mode");
  }

  // Set CUPTI parameters
  CUPTI_CHECK(cuptiActivitySetAttribute(
                  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                  cupti_activity_config::attr_value_size(
                      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE),
                  cupti_activity_config::attr_device_buffer_size()),
              err());
  CUPTI_CHECK(cuptiActivitySetAttribute(
                  CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                  cupti_activity_config::attr_value_size(
                      CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT),
                  cupti_activity_config::attr_device_buffer_pool_limit()),
              err());

  switch (mode_) {
  case Mode::ActivityTimeline:
    // Set handler function
    cupti_activity_config::set_activity_handler(tracing_activityHander);

    // disable preloads
    preload_nccl::set_passthrough(true);
    preload_cublas::set_passthrough(true);
    preload_cudnn::set_passthrough(true);

    // Enable CUPTI Activity API
    err() << "INFO: Profiler enabling activity API" << std::endl;
    for (const auto &kind : cuptiActivityKinds_) {
      err() << "DEBU: Enabling cuptiActivityKind " << kind << std::endl;
      CUptiResult code = cuptiActivityEnable(kind);
      if (code == CUPTI_ERROR_NOT_COMPATIBLE) {
        err() << "WARN: CUPTI_ERROR_NOT_COMPATIBLE when enabling " << kind
              << std::endl;
      } else {
        CUPTI_CHECK(code, err());
      }
    }
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(cuptiActivityBufferRequested,
                                               cuptiActivityBufferCompleted),
                err());

    break;
  case Mode::Full: {

    // Enable CUPTI Callback API
    profiler::err() << "INFO: CuptiSubscriber enabling callback API"
                    << std::endl;
    CUPTI_CHECK(cuptiSubscribe(&cuptiCallbackSubscriber_,
                               (CUpti_CallbackFunc)cuptiCallbackFunction,
                               nullptr),
                err());
    CUPTI_CHECK(cuptiEnableDomain(1, cuptiCallbackSubscriber_,
                                  CUPTI_CB_DOMAIN_RUNTIME_API),
                err());
    CUPTI_CHECK(cuptiEnableDomain(1, cuptiCallbackSubscriber_,
                                  CUPTI_CB_DOMAIN_DRIVER_API),
                err());
    profiler::err() << "INFO: done enabling callback API domains" << std::endl;

    // Enable CUPTI Activity API
    err() << "INFO: Profiler enabling activity API" << std::endl;
    for (const auto &kind : cuptiActivityKinds_) {
      err() << "DEBU: Enabling cuptiActivityKind " << kind << std::endl;
      CUptiResult code = cuptiActivityEnable(kind);
      if (code == CUPTI_ERROR_NOT_COMPATIBLE) {
        err() << "WARN: CUPTI_ERROR_NOT_COMPATIBLE when enabling " << kind
              << std::endl;
      } else if (code == CUPTI_ERROR_INVALID_KIND) {
        err() << "WARN: CUPTI_ERROR_INVALID_KIND when enabling " << kind
              << std::endl;
      } else {
        CUPTI_CHECK(code, err());
      }
    }
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(cuptiActivityBufferRequested,
                                               cuptiActivityBufferCompleted),
                err());
    profiler::err() << "INFO: done enabling activity API" << std::endl;
    break;
  }
  default: { assert(0 && "Unhandled mode"); }
  }

  err() << "INFO: scanning devices" << std::endl;
  hardware_.get_device_properties();
  err() << "INFO: done" << std::endl;

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

    zipkin::ZipkinOtTracerOptions overheadTracerOpts;
    overheadTracerOpts.service_name = "Profiler Overhead Tracer";
    overheadTracerOpts.collector_host = Profiler::instance().zipkin_host();
    overheadTracerOpts.collector_port = Profiler::instance().zipkin_port();
    overheadTracer_ = makeZipkinOtTracer(overheadTracerOpts);

    rootSpan_ = rootTracer_->StartSpan("Root");
  }

  isInitialized_ = true;
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}

static ProfilerInitializer pi;