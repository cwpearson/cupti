#include <cassert>
#include <cstdio>

#include "cprof/callbacks.hpp"
#include "cprof/profiler.hpp"
#include "util/environment_variable.hpp"

using namespace cprof;

/*! \brief Profiler() create a profiler
 *
 * Should not handle any initialization. Defer that to the init() method.
 */
Profiler::Profiler() : manager_(nullptr), isInitialized_(false) {
  printf("INFO: Profiler ctor\n");
  printf("INFO: done Profiler ctor\n");
}

Profiler::~Profiler() {
  std::cout << "INFO: Profiler dtor" << std::endl;
  delete manager_;
  isInitialized_ = false;
  printf("INFO: Profiler dtor done");
}

/*! \brief Profiler() initialize a profiler object
 *
 * Handle initialization here so that calls to Profiler member functions are
 * valid, since they've already been constructed.
 */
void Profiler::init() {
  printf("INFO: Profiler::init()\n");
  useCuptiCallback_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_CALLBACK", true).get();
  printf("INFO: useCuptiCallback: %d\n", useCuptiCallback_);

  useCuptiActivity_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_ACTIVITY", true).get();
  printf("INFO: useCuptiActivity: %d\n", useCuptiActivity_);

  jsonOutputPath_ =
      EnvironmentVariable<std::string>("CPROF_OUT", "output.cprof").get();
  printf("INFO: jsonOutputPath: %s\n", jsonOutputPath_.c_str());

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