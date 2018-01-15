#include <iostream>
#include <thread>
#include <vector>

#include "cupti_subscriber.hpp"
#include "kernel_time.hpp"
#include "profiler.hpp"

using profiler::err;

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

void CUPTIAPI CuptiSubscriber::cuptiActivityBufferRequested(
    uint8_t **buffer, size_t *size, size_t *maxNumRecords) {

  singleton->parent_span;

  uint8_t *rawBuffer;

  *size = BUFFER_SIZE * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = BUFFER_SIZE;

  if (*buffer == NULL) {
    profiler::err() << "ERROR: out of memory" << std::endl;
    exit(-1);
  }
}

static void handleCuptiKindKernel(CUpti_Activity *record) {
  err() << "INFO: handleCuptiKindKernel" << std::endl;
  auto kernelTimer = KernelCallTime::instance();
  auto kernel = (CUpti_ActivityKernel3 *)record;
  kernelTimer.kernel_activity_times(kernel->correlationId, kernel->start,
                                    kernel->end, kernel);
}

static void handleCuptiKindMemcpy(CUpti_Activity *record) {
  auto kernelTimer = KernelCallTime::instance();
  auto memcpyRecord = (CUpti_ActivityMemcpy *)record;
  kernelTimer.memcpy_activity_times(memcpyRecord);
}

void CUPTIAPI CuptiSubscriber::cuptiActivityBufferCompleted(CUcontext ctx,
                                                            uint32_t streamId,
                                                            uint8_t *buffer,
                                                            size_t size,
                                                            size_t validSize) {
  CUpti_Activity *record = NULL;

  std::cerr << "Empty buffer" << std::endl;
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
}