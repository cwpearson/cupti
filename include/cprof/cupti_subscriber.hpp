#ifndef CUPTI_SUBSCRIBER_HPP
#define CUPTI_SUBSCRIBER_HPP

#include <iostream>

#include "cprof/activity_callbacks.hpp"
#include "cprof/callbacks.hpp"
#include "cprof/kernel_time.hpp"
#include "cprof/util_cupti.hpp"

class CuptiSubscriber {
private:
  CUpti_SubscriberHandle subscriber_;
  CUpti_CallbackFunc callback_;
  bool enableZipkin_; ///< send traces to zipkin

public:
  zipkin::ZipkinOtTracerOptions options;
  zipkin::ZipkinOtTracerOptions memcpy_tracer_options;
  zipkin::ZipkinOtTracerOptions launch_tracer_options;

  std::shared_ptr<opentracing::Tracer> tracer;
  std::shared_ptr<opentracing::Tracer> memcpy_tracer;
  std::shared_ptr<opentracing::Tracer> launch_tracer;
  span_t parent_span;

  CuptiSubscriber(CUpti_CallbackFunc callback, const bool enableZipkin);
  void init();
  ~CuptiSubscriber();

  bool enable_zipkin() const { return enableZipkin_; }
};

#endif