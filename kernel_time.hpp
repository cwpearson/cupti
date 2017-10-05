#ifndef KERNEL_TIME_HPP
#define KERNEL_TIME_HPP

#include <cstdint>
#include <cupti.h>
#include <map>
#include <string>

#include "optional.hpp"
#include "thread.hpp"

typedef struct {
  uint64_t start_time;
  uint64_t end_time;
} time_points_t;

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
};

#endif