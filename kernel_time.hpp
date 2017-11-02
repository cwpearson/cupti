#ifndef KERNEL_TIME_HPP
#define KERNEL_TIME_HPP

#include <cstdint>
#include <cupti.h>
#include <map>
#include <string>

#include "optional.hpp"
#include "thread.hpp"
#include <zipkin/opentracing.h>

typedef struct {
  uint64_t start_time;
  uint64_t end_time;
} time_points_t;

typedef struct {
  cudaMemcpyKind memcpyType;
  size_t memcpySize;
} memcpy_info_t;

typedef std::unique_ptr<opentracing::v1::Span> span_t;

class KernelCallTime {
private:
  KernelCallTime();

public:
  void kernel_start_time(const CUpti_CallbackData *cbInfo);
  void kernel_end_time(const CUpti_CallbackData *cbInfo);
  void write_to_file();
  static KernelCallTime &instance();
  static std::map<uint32_t, time_points_t> tid_to_time;
  static std::map<uint32_t, const char *> correlation_to_function;
  static std::map<uint32_t, const char *> correlation_to_symbol;
  static std::map<uint32_t, std::chrono::time_point<std::chrono::system_clock> > correlation_to_start;
  static std::map<uint32_t, memcpy_info_t> correlation_id_to_info;

private: 
  char* memcpy_type_to_string(cudaMemcpyKind kind);
};

#endif