#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>

#include "cprof/callbacks.hpp"
#include "cprof/model/thread.hpp"
#include "cprof/profiler.hpp"
#include "util/environment_variable.hpp"

using namespace cprof;

/*! \brief Profiler() create a profiler
 *
 * Should not handle any initialization. Defer that to the init() method.
 */
Profiler::Profiler() : manager_(nullptr), isInitialized_(false) {}

Profiler::~Profiler() {
  logging::err() << "Profiler dtor\n";
  delete manager_;
  isInitialized_ = false;
  out().flush();
  logging::err() << "Profiler dtor almost done...\n";
  err().flush();
}

/*! \brief Profiler() initialize a profiler object
 *
 * Handle initialization here so that calls to Profiler member functions are
 * valid, since they've already been constructed.
 */
void Profiler::init() {
  std::cerr << model::get_thread_id() << std::endl;

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
  useCuptiCallback_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_CALLBACK", true).get();
  err() << "INFO: useCuptiCallback: " << useCuptiCallback_ << std::endl;

  useCuptiActivity_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_ACTIVITY", true).get();
  err() << "INFO: useCuptiActivity: " << useCuptiActivity_ << std::endl;

  zipkinHost_ =
      EnvironmentVariable<std::string>("CPROF_ZIPKIN_HOST", "localhost").get();
  err() << "INFO: zipkinEndpoint: " << zipkinHost_ << std::endl;

  zipkinPort_ = EnvironmentVariable<uint32_t>("CPROF_ZIPKIN_PORT", 9411u).get();
  err() << "INFO: zipkinPort: " << zipkinPort_ << std::endl;

  err() << "INFO: scanning devices" << std::endl;
  hardware_.get_device_properties();
  err() << "INFO: done" << std::endl;

  manager_ = new CuptiSubscriber((CUpti_CallbackFunc)callback);
  manager_->init();

  isInitialized_ = true;
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}

static ProfilerInitializer pi;