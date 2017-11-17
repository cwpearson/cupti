#include <iostream>

#include "cprof/activity_callbacks.hpp"
#include "cprof/callbacks.hpp"
#include "cprof/cupti_subscriber.hpp"
#include "cprof/kernel_time.hpp"
#include "cprof/util_cupti.hpp"

zipkin::ZipkinOtTracerOptions CuptiSubscriber::options;
zipkin::ZipkinOtTracerOptions CuptiSubscriber::memcpy_tracer_options;
zipkin::ZipkinOtTracerOptions CuptiSubscriber::launch_tracer_options;
std::shared_ptr<opentracing::Tracer> CuptiSubscriber::tracer;
std::shared_ptr<opentracing::Tracer> CuptiSubscriber::memcpy_tracer;
std::shared_ptr<opentracing::Tracer> CuptiSubscriber::launch_tracer;

span_t CuptiSubscriber::parent_span;

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
    printf("Error: out of memory\n");
    exit(-1);
  }
}

CuptiSubscriber::CuptiSubscriber(CUpti_CallbackFunc callback)
    : callback_(callback) {}

void CuptiSubscriber::init() {
  assert(callback_);
  printf("INFO: CuptiSubscriber init\n");

  size_t attrValueBufferSize = BUFFER_SIZE * 1024,
         attrValueSize = sizeof(size_t), attrValuePoolSize = BUFFER_SIZE;

  // Create tracers here so that they are not destroyed
  // when clearing buffer during destruction
  options.service_name = "Parent";
  memcpy_tracer_options.service_name = "Memory Copy";
  launch_tracer_options.service_name = "Kernel Launch";
  tracer = makeZipkinOtTracer(options);
  memcpy_tracer = makeZipkinOtTracer(memcpy_tracer_options);
  launch_tracer = makeZipkinOtTracer(launch_tracer_options);
  parent_span = tracer->StartSpan("Parent");

  cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                            &attrValueSize, &attrValueBufferSize);
  cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                            &attrValueSize, &attrValuePoolSize);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
  cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);

  printf("INFO: activating callbacks\n");
  CUPTI_CHECK(cuptiSubscribe(&subscriber_, callback_, nullptr));
  CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API));
  CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  printf("INFO: done activating callbacks\n");
}

CuptiSubscriber::~CuptiSubscriber() {
  printf("INFO: CuptiSubscriber dtor\n");
  cuptiActivityFlushAll(0);
  printf("INFO: done cuptiActivityFlushAll\n");
  parent_span->Finish();
  auto kernelTimer = KernelCallTime::instance();
  kernelTimer.flush_tracers();
  printf("Deactivating callbacks!\n");
  CUPTI_CHECK(cuptiUnsubscribe(subscriber_));
  printf("INFO: done deactivating callbacks!\n");
  printf("INFO: done CuptiSubscriber dtor\n");
}
