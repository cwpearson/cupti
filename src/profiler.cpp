#include <cassert>
#include <cstdio>

#include <fstream>
#include <iostream>
#include <memory>

#include "cprof/callbacks.hpp"
#include "cprof/profiler.hpp"
#include "util/environment_variable.hpp"

using namespace cprof;

/*! \brief Profiler() create a profiler
 *
 * Should not handle any initialization. Defer that to the init() method.
 */
Profiler::Profiler()
    : manager_(nullptr), err_(nullptr), out_(nullptr), isInitialized_(false) {}

Profiler::~Profiler() {
  delete manager_;
  isInitialized_ = false;
  if (out_) {
    out_->flush();
  }
  if (err_) {
    err_->flush();
  }
}

/*! \brief Profiler() initialize a profiler object
 *
 * Handle initialization here so that calls to Profiler member functions are
 * valid, since they've already been constructed.
 */
void Profiler::init() {
  auto outPath = EnvironmentVariable<std::string>("CPROF_OUT", "-").get();
  if (outPath != "-") {
    out_ = std::unique_ptr<std::ofstream>(
        new std::ofstream(outPath.c_str(), std::ios::app));
    assert(out_->good() && "Unable to open out file");
  }

  auto errPath = EnvironmentVariable<std::string>("CPROF_ERR", "-").get();
  if (errPath != "-") {
    err_ = std::unique_ptr<std::ofstream>(
        new std::ofstream(errPath.c_str(), std::ios::app));
    assert(err_->good() && "Unable to open err file");
  }

  err() << "INFO: Profiler::init()";
  useCuptiCallback_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_CALLBACK", true).get();
  printf("INFO: useCuptiCallback: %d\n", useCuptiCallback_);

  useCuptiActivity_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_ACTIVITY", true).get();
  printf("INFO: useCuptiActivity: %d\n", useCuptiActivity_);

  zipkinHost_ =
      EnvironmentVariable<std::string>("CPROF_ZIPKIN_HOST", "localhost").get();
  printf("INFO: zipkinEndpoint: %s\n", zipkinHost_.c_str());

  zipkinPort_ = EnvironmentVariable<uint32_t>("CPROF_ZIPKIN_PORT", 9411u).get();
  printf("INFO: zipkinPort: %u\n", zipkinPort_);

  printf("INFO: scanning devices\n");
  hardware_.get_device_properties();
  printf("INFO: done\n");

  manager_ = new CuptiSubscriber((CUpti_CallbackFunc)callback);
  manager_->init();

  isInitialized_ = true;
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}

static ProfilerInitializer pi;