#ifndef KERNEL_TIME_HPP
#define KERNEL_TIME_HPP

#include <cstdint>
#include <cupti.h>
#include <map>
#include <mutex>
#include <string>

#include "cprof/model/thread.hpp"
#include "text_map_carrier.hpp"
#include "util/optional.hpp"

typedef struct {
  uint64_t start_time;
  uint64_t end_time;
} time_points_t;

typedef struct {
  cudaMemcpyKind memcpyType;
  size_t memcpySize;
} memcpy_info_t;

class KernelCallTime {

private:
  std::mutex accessMutex_;

public:
  void kernel_start_time(const CUpti_CallbackData *cbInfo);
  void kernel_end_time(const CUpti_CallbackData *cbInfo);
  void kernel_activity_times(uint32_t cid, uint64_t startTime, uint64_t endTime,
                             CUpti_ActivityKernel3 *launchRecord);
  void memcpy_activity_times(CUpti_ActivityMemcpy *memcpyRecord);
  void save_configured_call(uint32_t cid, std::vector<uintptr_t> configCall);

  std::map<uint32_t, time_points_t> tid_to_time;
  std::map<uint32_t, const char *> correlation_to_function;
  std::map<uint32_t, const char *> correlation_to_symbol;
  std::map<uint32_t, std::chrono::time_point<std::chrono::system_clock>>
      correlation_to_start;
  std::map<uint32_t, memcpy_info_t> correlation_id_to_info;
  std::map<uint32_t, uintptr_t> correlation_to_dest;
  std::map<uintptr_t, TextMapCarrier> ptr_to_span;
  std::unordered_map<std::string, std::string> text_map;
  std::map<uint32_t, std::vector<uintptr_t>> cid_to_call;
  std::map<uint32_t, uint32_t> cid_to_tid;

   //First bool corresponds to CALLBACKS completion, second corresponds to ACTIVITY completion
   static std::map<uint32_t, std::tuple<bool, bool>> cid_to_completion;
   static std::map<uint32_t, std::map<std::string, std::string>> cid_to_values;
 
   void callback_add_annotations(uint32_t cid, std::map<std::string, std::string> callback_values);
   void activity_add_annotations(uint32_t cid, std::map<std::string, std::string> activity_values);
   void check_completion(uint32_t cid);
 

private:
  const char *memcpy_type_to_string(uint8_t kind);
};

class tidStats {
public:
  time_points_t time_point;
  const char *functionName;
  const char *symbolName;
  std::chrono::time_point<std::chrono::system_clock> start_point;
  memcpy_info_t memcpyInfo;
};

#endif
