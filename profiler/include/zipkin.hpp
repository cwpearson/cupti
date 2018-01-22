#ifndef PROFILER_ZIPKIN_HPP
#define PROFILER_ZIPKIN_HPP

#include <memory>

#include <zipkin/opentracing.h>

typedef std::unique_ptr<opentracing::v1::Span> span_t;

#endif