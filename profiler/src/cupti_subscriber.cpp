#include <iostream>

#include "cprof/util_cupti.hpp"

#include "cupti_activity.hpp"
#include "cupti_callback.hpp"
#include "cupti_subscriber.hpp"
#include "timer.hpp"
#include "profiler.hpp"

typedef void (*BufReqFun)(uint8_t **buffer, size_t *size,
                          size_t *maxNumRecords);

CuptiSubscriber::CuptiSubscriber(const bool enableActivityAPI,
                                 const bool enableCallbackAPI,
                                 const bool enableZipkin)
    : enableActivityAPI_(enableActivityAPI),
      enableCallbackAPI_(enableCallbackAPI), enableZipkin_(enableZipkin) {}

void CuptiSubscriber::init() {
  profiler::err() << "INFO: CuptiSubscriber init" << std::endl;

  size_t attrValueBufferSize = BUFFER_SIZE * 1024,
         attrValueSize = sizeof(size_t), attrValuePoolSize = BUFFER_SIZE;

  if (enableZipkin_) {
    profiler::err() << "INFO: CuptiSubscriber enable zipkin" << std::endl;
    // Create tracers here so that they are not destroyed
    // when clearing buffer during destruction
    options.service_name = "profiler";
    options.collector_host = Profiler::instance().zipkin_host();
    options.collector_port = Profiler::instance().zipkin_port();
    tracer = makeZipkinOtTracer(options);

    memcpy_tracer_options.service_name = "memcpy tracer";
    memcpy_tracer_options.collector_host = Profiler::instance().zipkin_host();
    memcpy_tracer_options.collector_port = Profiler::instance().zipkin_port();
    memcpy_tracer = makeZipkinOtTracer(memcpy_tracer_options);

    launch_tracer_options.service_name = "kernel tracer";
    launch_tracer_options.collector_host = Profiler::instance().zipkin_host();
    launch_tracer_options.collector_port = Profiler::instance().zipkin_port();
    launch_tracer = makeZipkinOtTracer(launch_tracer_options);

    overhead_tracer_options.service_name = "Profiler Overhead Tracer";
    overhead_tracer_options.collector_host = Profiler::instance().zipkin_host();
    overhead_tracer_options.collector_port = Profiler::instance().zipkin_port();
    overhead_tracer = makeZipkinOtTracer(overhead_tracer_options);

    parent_span = tracer->StartSpan("Parent");
  }

  if (enableActivityAPI_) {
    profiler::err() << "INFO: CuptiSubscriber enabling activity API"
                    << std::endl;
    cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE,
                              &attrValueSize, &attrValueBufferSize);
    cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
                              &attrValueSize, &attrValuePoolSize);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT);
    cuptiActivityRegisterCallbacks(cuptiActivityBufferRequested,
                                   cuptiActivityBufferCompleted);
    profiler::err() << "INFO: done registering activity callbacks" << std::endl;
  }

  if (enableCallbackAPI_) {
    profiler::err() << "INFO: CuptiSubscriber enabling callback API"
                    << std::endl;
    CUPTI_CHECK(cuptiSubscribe(&subscriber_,
                               (CUpti_CallbackFunc)cuptiCallbackFunction,
                               nullptr),
                profiler::err());
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API),
                profiler::err());
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API),
                profiler::err());
    profiler::err() << "INFO: done enabling callback API domains" << std::endl;
  }
}

CuptiSubscriber::~CuptiSubscriber() {
  profiler::err() << "INFO: CuptiSubscriber dtor" << std::endl;
  if (enableActivityAPI_) {
    profiler::err() << "INFO: CuptiSubscriber cleaning up activity API"
                    << std::endl;
    cuptiActivityFlushAll(0);
    profiler::err() << "INFO: done cuptiActivityFlushAll" << std::endl;
  }

  if (enableCallbackAPI_) {
    profiler::err() << "INFO: CuptiSubscriber Deactivating callback API!"
                    << std::endl;
    CUPTI_CHECK(cuptiUnsubscribe(subscriber_), profiler::err());
    profiler::err() << "INFO: done deactivating callbacks!" << std::endl;
  }

  if (enableZipkin_) {
    profiler::err() << "INFO: CuptiSubscriber finalizing Zipkin" << std::endl;
    parent_span->Finish();
    memcpy_tracer->Close();
    launch_tracer->Close();
  }

  profiler::err() << "INFO: done CuptiSubscriber dtor" << std::endl;
}
