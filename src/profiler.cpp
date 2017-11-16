#include <cassert>
#include <cstdio>

#include "cprof/profiler.hpp"

using namespace cprof;

/*! \brief Profiler() handles profiler initialization
 *
 */
Profiler::Profiler() {
  printf("INFO: Profiler ctor\n");

  printf("INFO: scanning devices\n");
  hardware_.get_device_properties();
  printf("INFO: done\n");
}

Profiler::~Profiler() { std::cout << "Profiler dtor" << std::endl; }

void Profiler::init() { instance(); }

Profiler &Profiler::instance() {
  static Profiler p;
  return p;
}
