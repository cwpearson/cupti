#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <ctime>
#include <cxxabi.h>
#include <inttypes.h>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
//Libraries required to write to JSON
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>


#include "cupti_subscriber.hpp"
#include "timer.hpp"
#include "profiler.hpp"

using namespace std::chrono;
using namespace zipkin;
using namespace opentracing;


void Timer::callback_add_annotations(const CUpti_CallbackData *cbInfo){
  uint64_t start;
  // CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start), std::cerr);

  span_t current_span;
  //To fill in with various data
}


/*CUDA8*/
void Timer::addKernelActivityAnnotations(CUpti_ActivityKernel3 *kernel_Activity){

  CUpti_ActivityKernel3 local_Kernel_Activity = *kernel_Activity;

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(local_Kernel_Activity.start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(local_Kernel_Activity.end);
  auto end_time_stamp =
    std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);

  span_t current_span;  
  current_span = Profiler::instance().manager_->launch_tracer->StartSpan(
    std::to_string(local_Kernel_Activity.correlationId),
    {
      FollowsFrom(&Profiler::instance().manager_->parent_span->context()),
      StartTimestamp(start_time_point)
    }
  );
  //Extract useful information from local_Kernel_Activity and add it to trace
  current_span->SetTag("blockX", std::to_string(local_Kernel_Activity.blockX));
  current_span->SetTag("blockY", std::to_string(local_Kernel_Activity.blockY));
  current_span->SetTag("blockZ", std::to_string(local_Kernel_Activity.blockZ));
  current_span->SetTag("completed", std::to_string(local_Kernel_Activity.completed));
  current_span->SetTag("deviceId", std::to_string(local_Kernel_Activity.deviceId));
  current_span->SetTag("dynamicSharedMemory", std::to_string(local_Kernel_Activity.dynamicSharedMemory));
  current_span->SetTag("gridId", std::to_string(local_Kernel_Activity.gridId));
  current_span->SetTag("gridX", std::to_string(local_Kernel_Activity.gridX));
  current_span->SetTag("gridY", std::to_string(local_Kernel_Activity.gridY));
  current_span->SetTag("gridZ", std::to_string(local_Kernel_Activity.gridZ));
  current_span->SetTag("localMemoryPerThread", std::to_string(local_Kernel_Activity.localMemoryPerThread));
  current_span->SetTag("localMemoryTotal", std::to_string(local_Kernel_Activity.localMemoryTotal));
  current_span->SetTag("registersPerThread", std::to_string(local_Kernel_Activity.registersPerThread));
  current_span->SetTag("sharedMemoryConfig", std::to_string(local_Kernel_Activity.sharedMemoryConfig));
  current_span->SetTag("staticSharedMemory", std::to_string(local_Kernel_Activity.staticSharedMemory));
  current_span->SetTag("streamId", std::to_string(local_Kernel_Activity.streamId));
  current_span->SetTag("staticSharedMemory", std::to_string(local_Kernel_Activity.staticSharedMemory));
  current_span->Finish({FinishTimestamp(end_time_stamp)});

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;
  
  ptree pt;

  pt.put("blockX", std::to_string(local_Kernel_Activity.blockX));
  pt.put("blockY", std::to_string(local_Kernel_Activity.blockY));
  pt.put("blockZ", std::to_string(local_Kernel_Activity.blockZ));
  pt.put("completed", std::to_string(local_Kernel_Activity.completed));
  pt.put("deviceId", std::to_string(local_Kernel_Activity.deviceId));
  pt.put("dynamicSharedMemory", std::to_string(local_Kernel_Activity.dynamicSharedMemory));
  pt.put("gridId", std::to_string(local_Kernel_Activity.gridId));
  pt.put("gridX", std::to_string(local_Kernel_Activity.gridX));
  pt.put("gridY", std::to_string(local_Kernel_Activity.gridY));
  pt.put("gridZ", std::to_string(local_Kernel_Activity.gridZ));
  pt.put("localMemoryPerThread", std::to_string(local_Kernel_Activity.localMemoryPerThread));
  pt.put("localMemoryTotal", std::to_string(local_Kernel_Activity.localMemoryTotal));
  pt.put("registersPerThread", std::to_string(local_Kernel_Activity.registersPerThread));
  pt.put("sharedMemoryConfig", std::to_string(local_Kernel_Activity.sharedMemoryConfig));
  pt.put("staticSharedMemory", std::to_string(local_Kernel_Activity.staticSharedMemory));
  pt.put("streamId", std::to_string(local_Kernel_Activity.streamId));
  pt.put("staticSharedMemory", std::to_string(local_Kernel_Activity.staticSharedMemory));
  std::ostringstream buf;
  write_json(buf, pt, false);
}

void Timer::addMemcpyActivityAnnotations(CUpti_ActivityMemcpy* memcpy_Activity){
  CUpti_ActivityMemcpy local_Memcpy_Activity = *memcpy_Activity;

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(local_Memcpy_Activity.start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(local_Memcpy_Activity.end);
  auto end_time_stamp =
    std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);

  span_t current_span;  
  current_span = Profiler::instance().manager_->memcpy_tracer->StartSpan(
    std::to_string(local_Memcpy_Activity.correlationId),
    {
      FollowsFrom(&Profiler::instance().manager_->parent_span->context()),
      StartTimestamp(start_time_point)
    }
  );
  current_span->SetTag("bytes", std::to_string(local_Memcpy_Activity.bytes));
  current_span->SetTag("contextId", std::to_string(local_Memcpy_Activity.contextId));
  current_span->SetTag("copyKind", std::to_string(local_Memcpy_Activity.copyKind));
  current_span->SetTag("deviceId", std::to_string(local_Memcpy_Activity.deviceId));
  current_span->SetTag("dstKind", std::to_string(local_Memcpy_Activity.dstKind));
  current_span->SetTag("flags", std::to_string(local_Memcpy_Activity.flags));
  current_span->SetTag("runtimeCorrelationId", std::to_string(local_Memcpy_Activity.runtimeCorrelationId));
  current_span->SetTag("srcKind", std::to_string(local_Memcpy_Activity.srcKind));
  current_span->SetTag("streamId", std::to_string(local_Memcpy_Activity.streamId));  
  current_span->Finish({FinishTimestamp(end_time_stamp)});

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;
  
  ptree pt;
  pt.put("bytes", std::to_string(local_Memcpy_Activity.bytes));
  pt.put("contextId", std::to_string(local_Memcpy_Activity.contextId));
  pt.put("copyKind", std::to_string(local_Memcpy_Activity.copyKind));
  pt.put("deviceId", std::to_string(local_Memcpy_Activity.deviceId));
  pt.put("dstKind", std::to_string(local_Memcpy_Activity.dstKind));
  pt.put("flags", std::to_string(local_Memcpy_Activity.flags));
  pt.put("runtimeCorrelationId", std::to_string(local_Memcpy_Activity.runtimeCorrelationId));
  pt.put("srcKind", std::to_string(local_Memcpy_Activity.srcKind));
  pt.put("streamId", std::to_string(local_Memcpy_Activity.streamId));  
  std::ostringstream buf;
  write_json(buf, pt, false);
}

void Timer::activity_add_annotations(CUpti_Activity * activity_data){
  switch(activity_data->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL: {
      auto activity_cast = (CUpti_ActivityKernel3 *)activity_data;
      addKernelActivityAnnotations(activity_cast);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      auto activity_cast = (CUpti_ActivityMemcpy *)activity_data;
      addMemcpyActivityAnnotations(activity_cast);
    } 
    default: {
      // assert(0 && "This shouldn't be happening...");
      auto activity_cast = activity_data;
    }
  };
}