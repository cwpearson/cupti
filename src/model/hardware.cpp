#include <cuda_runtime.h>

#include "cprof/model/hardware.hpp"
#include "cprof/util_cuda.hpp"

using namespace cprof::model;

void Hardware::get_device_properties() {
  static bool done = false;
  if (done)
    return;

  done = true;

  int numDevices;
  cudaGetDeviceCount(&numDevices);
  printf("INFO: scanning %d cuda devices\n", numDevices);
  for (int i = 0; i < numDevices; ++i) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
    cudaDevices_.push_back(cuda::Device(prop, i));
  }
}