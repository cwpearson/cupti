#include <boost/any.hpp>
#include <iostream>
#include <thread>
#include <vector>

#include "cupti_activity.hpp"
#include "profiler.hpp"
#include "timer.hpp"

using profiler::err;

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {

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

void threadFunc(uint8_t *localBuffer, size_t validSize) {
  CUpti_Activity *record = NULL;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    auto err = cuptiActivityGetNextRecord(localBuffer, validSize, &record);
    if (err == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }
    profiler::timer().activity_add_annotations(record);
  }
}

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {

  span_t activity_span;
  if (Profiler::instance().is_zipkin_enabled()) {
    activity_span = Profiler::instance().overheadTracer_->StartSpan(
        "Activity API",
        {FollowsFrom(&Profiler::instance().rootSpan_->context())});
  }
  // uint8_t *localBuffer;
  // localBuffer = (uint8_t *)malloc(BUFFER_SIZE * 1024 + ALIGN_SIZE);
  // ALIGN_BUFFER(localBuffer, ALIGN_SIZE);
  // memcpy(localBuffer, buffer, validSize);
  // threadFunc()
  threadFunc(buffer, validSize);
  if (Profiler::instance().is_zipkin_enabled()) {
    activity_span->Finish();
  }
}