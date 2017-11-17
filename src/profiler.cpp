#include <cassert>
#include <cstdio>

#include "cprof/profiler.hpp"
#include "cprof/callbacks.hpp"

using namespace cprof;

/*! \brief Profiler() handles profiler initialization
 *
 */
Profiler::Profiler() {
  printf("INFO: Profiler ctor\n");

  printf("INFO: scanning devices\n");
  hardware_.get_device_properties();
  printf("INFO: done\n");

  // CuptiSubscriber Manager((CUpti_CallbackFunc)callback);
  manager_ = new CuptiSubscriber((CUpti_CallbackFunc)callback);
  manager_->init();
    printf("INFO: done Profiler ctor\n");
}

Profiler::~Profiler() { std::cout << "INFO: Profiler dtor" << std::endl;
delete manager_;
printf("INFO: Profiler dtor done"); }

void Profiler::init() { 
  printf("INFO: Profiler::init()\n");
  instance();
}

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}


static ProfilerInitializer pi;