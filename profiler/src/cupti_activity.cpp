#include <boost/any.hpp>
#include <iostream>
#include <thread>
#include <vector>

#include "cupti_activity.hpp"
#include "profiler.hpp"
#include "timer.hpp"

using profiler::err;

namespace cupti_activity_config {
size_t bufferSize;
size_t attrValueBufferSize;
size_t attrValuePoolSize;
size_t attrValueSize;

size_t alignSize = 8;
void set_buffer_size(const size_t n) {
  bufferSize = n;
  attrValueBufferSize = bufferSize * 1024;
  attrValueSize = sizeof(size_t);
  attrValuePoolSize = bufferSize;
}
size_t buffer_size() { return bufferSize; }
size_t align_size() { return alignSize; }

size_t *attr_value_buffer_size() { return &attrValueBufferSize; }
size_t *attr_value_size() { return &attrValueSize; }
size_t *attr_value_pool_size() { return &attrValuePoolSize; }

} // namespace cupti_activity_config

#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {

  using cupti_activity_config::align_size;
  using cupti_activity_config::buffer_size;

  uint8_t *rawBuffer;

  *size = buffer_size() * 1024;
  rawBuffer = (uint8_t *)malloc(*size + align_size());

  *buffer = ALIGN_BUFFER(rawBuffer, align_size());
  *maxNumRecords = cupti_activity_config::bufferSize;

  if (*buffer == NULL) {
    profiler::err() << "ERROR: out of memory" << std::endl;
    exit(-1);
  }
}

void threadFunc(uint8_t *localBuffer, size_t validSize) {

  using cupti_activity_config::buffer_size;

  CUpti_Activity *record = NULL;
  for (size_t i = 0; i < buffer_size(); i++) {
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