#include <cassert>
#include <cstdio>

#include "cprof/callbacks.hpp"
#include "cprof/profiler.hpp"
#include "util/environment_variable.hpp"

using namespace cprof;

/*! \brief Profiler() handles profiler initialization
 *
 */
Profiler::Profiler() {
  printf("INFO: Profiler ctor\n");

  useCuptiCallback_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_CALLBACK", true).get();
  printf("INFO: useCuptiCallback: %d\n", useCuptiCallback_);

  useCuptiActivity_ =
      EnvironmentVariable<bool>("CPROF_USE_CUPTI_ACTIVITY", true).get();
  printf("INFO: useCuptiActivity: %d\n", useCuptiActivity_);

  jsonOutputPath_ =
      EnvironmentVariable<std::string>("CPROF_OUT", "output.cprof").get();
  printf("INFO: jsonOutputPath: %s\n", jsonOutputPath_.c_str());

  zipkinEndpoint_ =
      EnvironmentVariable<std::string>("CPROF_ZIPKIN_ENDPOINT", "localhost")
          .get();
  printf("INFO: zipkinEndpoint: %s\n", zipkinEndpoint_.c_str());

  printf("INFO: scanning devices\n");
  hardware_.get_device_properties();
  printf("INFO: done\n");

  // CuptiSubscriber Manager((CUpti_CallbackFunc)callback);
  manager_ = new CuptiSubscriber((CUpti_CallbackFunc)callback);
  manager_->init();
  printf("INFO: done Profiler ctor\n");
}

Profiler::~Profiler() {
  std::cout << "INFO: Profiler dtor" << std::endl;
  delete manager_;
  printf("INFO: Profiler dtor done");
}

void Profiler::init() {
  printf("INFO: Profiler::init()\n");
  instance();
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}

static ProfilerInitializer pi;