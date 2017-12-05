#include <iostream>
#include <thread>
#include <vector>

#include "activity_callbacks.hpp"
#include "kernel_time.hpp"
#include "profiler.hpp"

using profiler::err;

void handleCuptiKindKernel(CUpti_Activity *record) {
  err() << "INFO: handleCuptiKindKernel" << std::endl;
  auto kernelTimer = KernelCallTime::instance();
  auto kernel = (CUpti_ActivityKernel3 *)record;
  kernelTimer.kernel_activity_times(kernel->correlationId, kernel->start,
                                    kernel->end, kernel);
}

void handleCuptiKindMemcpy(CUpti_Activity *record) {
  auto kernelTimer = KernelCallTime::instance();
  auto memcpyRecord = (CUpti_ActivityMemcpy *)record;
  kernelTimer.memcpy_activity_times(memcpyRecord);
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
  CUpti_Activity *record = NULL;

  std::cout << "Empty buffer" << std::endl;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    auto err = cuptiActivityGetNextRecord(buffer, validSize, &record);

    if (err == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }

    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
      handleCuptiKindKernel(record);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      handleCuptiKindMemcpy(record);
      break;
    default:
      exit(-1);
    }
  }
  auto kernelTimer = KernelCallTime::instance();
  kernelTimer.flush_tracers();
  //   free(buffer);
}