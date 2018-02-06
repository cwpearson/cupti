#include <iostream>
#include <thread>
#include <vector>
#include <boost/any.hpp>

#include "cupti_subscriber.hpp"
#include "timer.hpp"
#include "profiler.hpp"

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

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {

  uint8_t *localBuffer;  
  localBuffer = (uint8_t *)malloc(BUFFER_SIZE * 1024 + ALIGN_SIZE);
  ALIGN_BUFFER(localBuffer, ALIGN_SIZE);
  memcpy(localBuffer, buffer, validSize);
  Profiler::instance().activity_cleanup(localBuffer, validSize);
}