#ifndef CUPTI_SUBSCRIBER_HPP
#define CUPTI_SUBSCRIBER_HPP


#include <iostream>

#include "cprof/callbacks.hpp"
#include "cprof/activity_callbacks.hpp"
#include "cprof/util_cupti.hpp"
#include "cprof/kernel_time.hpp"


class CuptiSubscriber {
private:
  CUpti_SubscriberHandle subscriber_;

public:
  static zipkin::ZipkinOtTracerOptions options; 
  static zipkin::ZipkinOtTracerOptions memcpy_tracer_options;
  static zipkin::ZipkinOtTracerOptions launch_tracer_options;   

  static std::shared_ptr<opentracing::Tracer> tracer;
  static std::shared_ptr<opentracing::Tracer> memcpy_tracer;
  static std::shared_ptr<opentracing::Tracer> launch_tracer;
  static span_t parent_span;  
  

  CuptiSubscriber(CUpti_CallbackFunc callback);

  ~CuptiSubscriber();

};

#endif