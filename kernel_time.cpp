#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>

#include "kernel_time.hpp"

using namespace std::chrono;

std::map<uint32_t, time_points_t> KernelCallTime::tid_to_time;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_function;

KernelCallTime &KernelCallTime::instance() {
  static KernelCallTime a;
  return a;
}

KernelCallTime::KernelCallTime() {}

void KernelCallTime::kernel_start_time(const CUpti_CallbackData *cbInfo) {
  uint64_t startTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &startTimeStamp);
  time_points_t time_point;
  time_point.start_time = startTimeStamp;
  printf("-- %d -- The start time stamp is %ul\n", cbInfo->correlationId, startTimeStamp);
  
  this->tid_to_time.insert(
      std::pair<uint32_t, time_points_t>(cbInfo->correlationId, time_point));
  this->correlation_to_function.insert(std::pair<uint32_t, const char *>(
      cbInfo->correlationId, cbInfo->functionName));

            
}

void KernelCallTime::kernel_end_time(const CUpti_CallbackData *cbInfo) {
  uint64_t endTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
  auto time_point_iterator = this->tid_to_time.find(cbInfo->correlationId);
  time_point_iterator->second.end_time = endTimeStamp;
  printf("-- %d -- The end time stamp is %ul\n", cbInfo->correlationId, endTimeStamp);
}

void KernelCallTime::write_to_file() {
  printf("KernelCallTime write to file\n");

  for (auto iter = this->tid_to_time.begin(); iter != this->tid_to_time.end();
       iter++) {
    printf("%s - Time - %ul ns\n", this->correlation_to_function.find(iter->first)->second, iter->second.end_time - iter->second.start_time);
  }
}