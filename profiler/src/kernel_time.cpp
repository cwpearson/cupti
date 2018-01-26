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

#include "cupti_subscriber.hpp"
#include "kernel_time.hpp"
#include "profiler.hpp"

using namespace std::chrono;
using namespace zipkin;
using namespace opentracing;

static std::unordered_map<std::string, std::string> text_map;
static TextMapCarrier carrier(text_map);

std::map<uint32_t, std::tuple<bool, bool>> KernelCallTime::cid_to_completion;
std::map<uint32_t, std::map<std::string, std::string>> KernelCallTime::cid_to_values;

void KernelCallTime::kernel_start_time(const CUpti_CallbackData *cbInfo) {
  // std::lock_guard<std::mutex> guard(accessMutex_);
  // uint64_t startTimeStamp;
  // cuptiDeviceGetTimestamp(cbInfo->context, &startTimeStamp);
  // const char *cudaMem = "cudaMemcpy";
  // auto correlationId = cbInfo->correlationId;
  // time_points_t time_point;
  // time_point.start_time = startTimeStamp;
  // const char *memcpy = "cudaMemcpy";

  // if (strcmp(cbInfo->functionName, memcpy) == 0) {
  //   auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
  //   memcpy_info_t memInfo;

  //   this->correlation_to_dest.insert(
  //       std::pair<uint32_t, uintptr_t>(correlationId, (uintptr_t)params->dst));

  //   memInfo.memcpyType = params->kind;
  //   memInfo.memcpySize = params->count;
  //   this->correlation_id_to_info.insert(
  //       std::pair<uint32_t, memcpy_info_t>(correlationId, memInfo));
  // }

  // auto t1 = SystemClock::now();
  // this->correlation_to_start.insert(
  //     std::pair<uint32_t, std::chrono::time_point<std::chrono::system_clock>>(
  //         correlationId, t1));

  // // this->tid_to_time.insert(
  // // std::pair<uint32_t, time_points_t>(cbInfo->correlationId, time_point));
  // this->correlation_to_function.insert(std::pair<uint32_t, const char *>(
  //     cbInfo->correlationId, cbInfo->functionName));
  // this->correlation_to_symbol.insert(std::pair<uint32_t, const char *>(
  //     cbInfo->correlationId, cbInfo->symbolName));
  // this->cid_to_tid.insert(std::pair<uint32_t, uint32_t>(
  //     correlationId, cprof::model::get_thread_id()));
}

void KernelCallTime::kernel_end_time(const CUpti_CallbackData *cbInfo) {
  // uint64_t endTimeStamp;
  // cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
  // auto correlationId = cbInfo->correlationId;
  // auto t2 = SystemClock::now();
  // auto t1 = this->correlation_to_start.find(correlationId)->second;
  // auto milliseconds =
  // std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1); auto ms =
  // milliseconds.count();
  // // std::cout << ms << "ms" << std::endl;

  // span_t current_span;
  // const char *launchVal = "cudaLaunch";

  // ZipkinOtTracerOptions options;
  // options.service_name = "Kernel Launch " + std::to_string(correlationId);

  // bool found = false;

  // if (strcmp(cbInfo->functionName, launchVal) == 0){
  //   return;
  //   // for (auto iter = this->correlation_to_dest.begin(); iter !=
  //   this->correlation_to_dest.end(); iter++) {
  //   //   for (size_t argIdx = 0; argIdx < ConfiguredCall().args.size();
  //   ++argIdx) {
  //   //     if (ConfiguredCall().args[argIdx] == iter->second){
  //   //       // Dependency exists!
  //   //       auto spanIter =
  //   this->ptr_to_span.find(ConfiguredCall().args[argIdx]);
  //   //       if (spanIter != this->ptr_to_span.end()){

  //   //         auto span_context_maybe = launch_tracer->Extract(carrier);
  //   //         assert(span_context_maybe);
  //   //         auto contextPtr = spanIter->second;
  //   //         if (!found){
  //   //           current_span =
  //   launch_tracer->StartSpan(std::to_string(correlationId),
  //   //           {FollowsFrom(span_context_maybe->get()),
  //   StartTimestamp(t1)});
  //   //           found = true;
  //   //           break;
  //   //         } else {
  //   //           break;
  //   //         }
  //   //       }
  //   //     }
  //   //   }
  //   // }
  // } else {
  //   current_span = memcpy_tracer->StartSpan(std::to_string(correlationId),
  // //                     {ChildOf(&parent_span->context()),
  // StartTimestamp(t1)});

  //   auto iter = this->correlation_to_dest.find(correlationId);

  //   auto err = tracer->Inject(current_span->context(), carrier);
  //   assert(err);

  // this->ptr_to_cid.insert(std::pair<uintptr_t, TextMapCarrier>(iter->second,
  // carrier));
  // }
  // auto iter = this->correlation_to_function.find(correlationId);

  // if (iter != this->correlation_to_function.end() && iter->second != NULL){
  //   current_span->SetTag(
  //     "Function Name",
  //     iter->second);
  // }

  // auto cudaContext = cbInfo->context;
  // uint32_t deviceID;

  // cuptiGetDeviceId(cudaContext, &deviceID);
  // current_span->SetTag("Current device", std::to_string(deviceID));

  //   auto cStrSymbol =
  //   this->correlation_to_symbol.find(correlationId)->second; if (cStrSymbol
  //   != NULL) {
  //     current_span->SetTag("Symbol ", cStrSymbol);
  //   }

  // if (this->correlation_id_to_info.find(correlationId) !=
  //     this->correlation_id_to_info.end()) {
  //   auto memCpyIter = this->correlation_id_to_info.find(correlationId);
  //   auto memCpyInfo = memCpyIter->second;
  //   current_span->SetTag("Transfer size", memCpyInfo.memcpySize);
  //   current_span->SetTag("Transfer type",
  //                        memcpy_type_to_string(memCpyInfo.memcpyType));
  // }

  // current_span->Finish();

  // auto time_point_iterator = this->tid_to_time.find(cbInfo->correlationId);
  // // time_point_iterator->second.end_time = endTimeStamp;
  // auto time_point = time_point_iterator->second;
  // // std::cout << "Start time " << time_point.start_time << std::endl;
  // // std::cout << "End time " << time_point.end_time << std::endl;
  // if (time_point_iterator != this->tid_to_time.end())
  //   std::cout <<  "Function Name " << cbInfo->functionName << " Time from
  //   cuda " << time_point.end_time - time_point.start_time << std::endl;
}

const char *KernelCallTime::memcpy_type_to_string(uint8_t kind) {
  switch (kind) {
  case cudaMemcpyHostToHost: {
    return "cudaMemcpyHostToHost";
  }
  case cudaMemcpyHostToDevice: {
    return "cudaMemcpyHostToDevice";
  }
  case cudaMemcpyDeviceToHost: {
    return "cudaMemcpyDeviceToHost";
  }
  case cudaMemcpyDeviceToDevice: {
    return "cudaMemcpyDeviceToDevice";
  }
  case cudaMemcpyDefault: {
    return "cudaMemcpyDefault";
  }
  default: { return "Unknown"; }
  }
}

void KernelCallTime::memcpy_activity_times(CUpti_ActivityMemcpy *memcpyRecord) {
  if (Profiler::instance().manager_->enable_zipkin()) {
    span_t current_span;
    auto correlationId = memcpyRecord->correlationId;
    auto tidIter = this->cid_to_tid.find(correlationId);
    int threadId = -1;
    if (tidIter != this->cid_to_tid.end()) {
      threadId = tidIter->second;
    }

    std::chrono::nanoseconds start_dur(memcpyRecord->start);
    std::chrono::nanoseconds end_dur(memcpyRecord->end);

    auto start_time_point =
        std::chrono::duration_cast<std::chrono::nanoseconds>(start_dur);
    auto end_time_point =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);

    // auto dependency_tracking = DependencyTracking::instance();
    // dependency_tracking.annotate_times(correlationId, start_time_point, end_time_point);

    current_span = Profiler::instance().manager_->memcpy_tracer->StartSpan(
        std::to_string(correlationId),
        {ChildOf(&Profiler::instance().manager_->parent_span->context()),
         StartTimestamp(start_time_point)});

    current_span->SetTag("Transfer size", memcpyRecord->bytes);
    current_span->SetTag("Transfer type",
                         memcpy_type_to_string(memcpyRecord->copyKind));
    current_span->SetTag("Host Thread", std::to_string(threadId));

    auto timeElapsed = memcpyRecord->end - memcpyRecord->start;
    current_span->SetTag("CUPTI Duration", std::to_string(timeElapsed));
    // auto err = tracer->Inject(current_span->context(), carrier);
    current_span->Finish({FinishTimestamp(end_time_point)});

    // assert(err);

    auto iter = this->correlation_to_dest.find(correlationId);
    this->ptr_to_span.insert(
        std::pair<uintptr_t, TextMapCarrier>(iter->second, carrier));
  }
}

void KernelCallTime::kernel_activity_times(
    uint32_t cid, uint64_t startTime, uint64_t endTime,
    CUpti_ActivityKernel3 *launchRecord) {
  if (Profiler::instance().manager_->enable_zipkin()) {
    auto found = false;
    span_t current_span;
    auto correlationId = cid;

    auto tidIter = this->cid_to_tid.find(correlationId);
    int threadId = -1;
    if (tidIter != this->cid_to_tid.end()) {
      threadId = tidIter->second;
    }

    std::chrono::nanoseconds start_dur(startTime);
    std::chrono::nanoseconds end_dur(endTime);

    auto start_time_point =
        std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
    auto end_time_point =
        std::chrono::duration_cast<std::chrono::microseconds>(end_dur);

    // auto dependency_tracking = DependencyTracking::instance();
    // dependency_tracking.annotate_times(correlationId, start_time_point, end_time_point);

    auto configCallIter = this->cid_to_call.find(correlationId);
    // if (cid_to_call.end() != configCallIter){
    //   std::cout << "Here!" << std::endl;
    //   auto configCall = configCallIter->second;

    //   for (auto iter = this->correlation_to_dest.begin(); iter !=
    //   this->correlation_to_dest.end(); iter++) {
    //       for (size_t argIdx = 0; argIdx < configCall.size(); ++argIdx) {
    //         if (configCall[argIdx] == iter->second){
    //           // Dependency exists!
    //           auto spanIter = this->ptr_to_span.find(configCall[argIdx]);
    //           if (spanIter != this->ptr_to_span.end()){

    //             std::cout << "we baking" << std::endl;
    //             auto span_context_maybe = launch_tracer->Extract(carrier);
    //             assert(span_context_maybe);
    //             auto contextPtr = spanIter->second;
    //             if (!found){
    //                 current_span =
    //                 launch_tracer->StartSpan(std::to_string(correlationId),
    //                   {FollowsFrom(span_context_maybe->get()),
    //                   StartTimestamp(start_time_point)});

    //               current_span->SetTag(
    //                 "Function Name",
    //                 "I'm ugly");
    //               found = true;
    //               break;
    //             }
    //           }
    //         }
    //       }
    //     }
    // } else
    if (!found) {
      current_span = Profiler::instance().manager_->launch_tracer->StartSpan(
          std::to_string(correlationId),
          {FollowsFrom(&Profiler::instance().manager_->parent_span->context()),
           StartTimestamp(start_time_point)});
    }

    current_span->SetTag("Function Name", "cudaLaunch");

    auto timeElapsed = endTime - startTime;
    current_span->SetTag("CUPTI Duration", std::to_string(timeElapsed));

    current_span->SetTag("Current device",
                         std::to_string(launchRecord->deviceId));
    current_span->SetTag("Host Thread", std::to_string(threadId));
    //   auto cStrSymbol =
    //   this->correlation_to_symbol.find(correlationId)->second; if (cStrSymbol
    //   != NULL) {
    current_span->SetTag("Name ", launchRecord->name);
    //   }

    current_span->Finish({FinishTimestamp(end_time_point)});
  }
}

void KernelCallTime::save_configured_call(uint32_t cid,
                                          std::vector<uintptr_t> configCall) {
  this->cid_to_call.insert(
      std::pair<uint32_t, std::vector<uintptr_t>>(cid, configCall));
}

void KernelCallTime::callback_add_annotations(uint32_t cid, std::map<std::string, std::string> callback_values){
  for (auto value : callback_values){
    std::cout << "Callback add annotations" << std::endl;
    //Create associated span for the callback
    // dataValues->second.insert(value);
  }
}


/*CUDA8*/
void addKernelActivityAnnotations(CUpti_ActivityKernel3 *kernel_Activity){

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
}

void addMemcpyActivityAnnotations(CUpti_ActivityMemcpy* memcpy_Activity){
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
}

void KernelCallTime::activity_add_annotations(CUpti_Activity * activity_data){
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