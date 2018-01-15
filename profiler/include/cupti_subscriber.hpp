#ifndef CUPTI_SUBSCRIBER_HPP
#define CUPTI_SUBSCRIBER_HPP

#include <iostream>

#include "cprof/util_cupti.hpp"

#include "activity_callbacks.hpp"
#include "cupti_callbacks.hpp"
#include "kernel_time.hpp"

class CuptiSubscriber {
private:
  CUpti_SubscriberHandle subscriber_;
  CUpti_CallbackFunc callback_;
  bool enableActivityAPI_; ///< gather info through the CUPTI activity API
  bool enableCallbackAPI_; ///< gather info from the CUPTI callback API
  bool enableZipkin_;      ///< send traces to zipkin

public:
  zipkin::ZipkinOtTracerOptions options;
  zipkin::ZipkinOtTracerOptions memcpy_tracer_options;
  zipkin::ZipkinOtTracerOptions launch_tracer_options;

  std::shared_ptr<opentracing::Tracer> tracer;
  std::shared_ptr<opentracing::Tracer> memcpy_tracer;
  std::shared_ptr<opentracing::Tracer> launch_tracer;
  span_t parent_span;

  CuptiSubscriber(CUpti_CallbackFunc callback, const bool enableActivityAPI,
                  const bool enableCallbackAPI, const bool enableZipkin);
  void init();
  ~CuptiSubscriber();

  bool enable_zipkin() const { return enableZipkin_; }
};

#endif