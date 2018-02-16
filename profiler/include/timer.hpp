#ifndef KERNEL_TIME_HPP
#define KERNEL_TIME_HPP

#include <cstdint>
#include <cupti.h>
#include <map>
#include <mutex>
#include <string>
#include <boost/lexical_cast.hpp>
#include <boost/any.hpp>

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

class Timer {

private:
  std::mutex accessMutex_;

public:
   void callback_add_annotations(const CUpti_CallbackData *cbInfo, CUpti_CallbackId cbid);
   void activity_add_annotations(CUpti_Activity * activity_data);
 

private:
  const char *memcpy_type_to_string(uint8_t kind);
  void addKernelActivityAnnotations(CUpti_ActivityKernel3 *kernel_Activity);
  void addMemcpyActivityAnnotations(CUpti_ActivityMemcpy* memcpy_Activity);
  void addEnvironmentActivityAnnotations(CUpti_ActivityEnvironment* environment_Activity);
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
