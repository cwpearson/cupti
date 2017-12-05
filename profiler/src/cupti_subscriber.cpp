#include <iostream>

#include "cprof/util_cupti.hpp"

#include "activity_callbacks.hpp"
#include "cupti_callbacks.hpp"
#include "cupti_subscriber.hpp"
#include "kernel_time.hpp"
#include "profiler.hpp"

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
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

CuptiSubscriber::CuptiSubscriber(CUpti_CallbackFunc callback,
                                 const bool enableZipkin)
    : callback_(callback), enableZipkin_(enableZipkin) {}

void CuptiSubscriber::init() {
  assert(callback_);
  profiler::err() << "INFO: CuptiSubscriber init" << std::endl;

  size_t attrValueBufferSize = BUFFER_SIZE * 1024,
         attrValueSize = sizeof(size_t), attrValuePoolSize = BUFFER_SIZE;

  if (enableZipkin_) {
    // Create tracers here so that they are not destroyed
    // when clearing buffer during destruction
    options.service_name = "Parent";
    memcpy_tracer_options.service_name = "Memory Copy";
    launch_tracer_options.service_name = "Kernel Launch";
    options.collector_host = Profiler::instance().zipkin_host();
    memcpy_tracer_options.collector_host = Profiler::instance().zipkin_host();
    launch_tracer_options.collector_host = Profiler::instance().zipkin_host();
    options.collector_port = Profiler::instance().zipkin_port();
    memcpy_tracer_options.collector_port = Profiler::instance().zipkin_port();
    launch_tracer_options.collector_port = Profiler::instance().zipkin_port();
    memcpy_tracer = makeZipkinOtTracer(memcpy_tracer_options);
    tracer = makeZipkinOtTracer(options);
    launch_tracer = makeZipkinOtTracer(launch_tracer_options);
    parent_span = tracer->StartSpan("Parent");
  }

  cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                            &attrValueSize, &attrValueBufferSize);
  cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                            &attrValueSize, &attrValuePoolSize);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);

  profiler::err() << "INFO: activating callbacks" << std::endl;
  CUPTI_CHECK(cuptiSubscribe(&subscriber_, callback_, nullptr),
              profiler::err());
  CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API),
              profiler::err());
  CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API),
              profiler::err());
  profiler::err() << "INFO: done activating callbacks" << std::endl;
}

CuptiSubscriber::~CuptiSubscriber() {
  profiler::err() << "INFO: CuptiSubscriber dtor" << std::endl;
  cuptiActivityFlushAll(0);
  profiler::err() << "INFO: done cuptiActivityFlushAll" << std::endl;
  if (enableZipkin_) {
    parent_span->Finish();
  }
  auto kernelTimer = KernelCallTime::instance();
  kernelTimer.flush_tracers();
  profiler::err() << "Deactivating callbacks!" << std::endl;
  CUPTI_CHECK(cuptiUnsubscribe(subscriber_), profiler::err());
  profiler::err() << "INFO: done deactivating callbacks!" << std::endl;
  profiler::err() << "INFO: done CuptiSubscriber dtor" << std::endl;
}
