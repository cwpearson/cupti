#ifndef KERNEL_TIME_HPP
#define KERNEL_TIME_HPP

#include <cstdint>
#include <cupti.h>
#include <map>
#include <string>

#include "callbacks.hpp"
#include "optional.hpp"
#include "thread.hpp"
#include "text_map_carrier.hpp"
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
// typedef std::unique_ptr<zipkin::Span> span_t;

class KernelCallTime {
private:
  KernelCallTime();

public:
  void kernel_start_time(const CUpti_CallbackData *cbInfo);
  void kernel_end_time(const CUpti_CallbackData *cbInfo);
  void kernel_activity_times(uint32_t cid, uint64_t startTime, uint64_t endTime, CUpti_ActivityKernel3* launchRecord);
  void memcpy_activity_times(CUpti_ActivityMemcpy * memcpyRecord);
  void save_configured_call(uint32_t cid, std::vector<uintptr_t> configCall);
  void write_to_file();
  static KernelCallTime &instance();

  static std::map<uint32_t, time_points_t> tid_to_time;
  static std::map<uint32_t, const char *> correlation_to_function;
  static std::map<uint32_t, const char *> correlation_to_symbol;
  static std::map<uint32_t, std::chrono::time_point<std::chrono::system_clock> > correlation_to_start;
  static std::map<uint32_t, memcpy_info_t> correlation_id_to_info;
  static std::map<uint32_t, uintptr_t>  correlation_to_dest;
  static std::map<uintptr_t, TextMapCarrier> ptr_to_span;
  static std::unordered_map<std::string, std::string> text_map;  
  static std::map<uint32_t, std::vector<uintptr_t> > cid_to_call;
  
private: 
  char* memcpy_type_to_string(uint8_t kind);
};

class tidStats {
public:
  time_points_t time_point;
  const char* functionName;
  const char* symbolName;
  std::chrono::time_point<std::chrono::system_clock> start_point;
  memcpy_info_t memcpyInfo;

private:

};

#endif