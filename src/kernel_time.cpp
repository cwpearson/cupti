#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <ctime>
#include <cxxabi.h>
#include <inttypes.h>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "cprof/callbacks.hpp"
#include "cprof/cupti_subscriber.hpp"
#include "cprof/kernel_time.hpp"
#include "cprof/profiler.hpp"

using namespace std::chrono;
using namespace zipkin;
using namespace opentracing;
using cprof::Profiler;

std::map<uint32_t, time_points_t> KernelCallTime::tid_to_time;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_function;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_symbol;
std::map<uint32_t, std::chrono::time_point<std::chrono::system_clock>>
    KernelCallTime::correlation_to_start;
std::map<uint32_t, memcpy_info_t> KernelCallTime::correlation_id_to_info;
std::map<uint32_t, uintptr_t> KernelCallTime::correlation_to_dest;
std::map<uintptr_t, TextMapCarrier> KernelCallTime::ptr_to_span;
std::unordered_map<std::string, std::string> KernelCallTime::text_map;
std::map<uint32_t, std::vector<uintptr_t>> KernelCallTime::cid_to_call;
std::map<uint32_t, uint32_t> KernelCallTime::cid_to_tid;

static std::unordered_map<std::string, std::string> text_map;
static TextMapCarrier carrier(text_map);

std::mutex kctMutex_;

KernelCallTime &KernelCallTime::instance() {
  static KernelCallTime a;
  return a;
}

KernelCallTime::KernelCallTime() {}

void KernelCallTime::kernel_start_time(const CUpti_CallbackData *cbInfo) {
  std::lock_guard<std::mutex> guard(kctMutex_);
  uint64_t startTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &startTimeStamp);
  const char *cudaMem = "cudaMemcpy";
  auto correlationId = cbInfo->correlationId;
  time_points_t time_point;
  time_point.start_time = startTimeStamp;
  const char *memcpy = "cudaMemcpy";

  if (strcmp(cbInfo->functionName, memcpy) == 0) {
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    memcpy_info_t memInfo;

    this->correlation_to_dest.insert(
        std::pair<uint32_t, uintptr_t>(correlationId, (uintptr_t)params->dst));

    memInfo.memcpyType = params->kind;
    memInfo.memcpySize = params->count;
    this->correlation_id_to_info.insert(
        std::pair<uint32_t, memcpy_info_t>(correlationId, memInfo));
  }

  auto t1 = SystemClock::now();
  this->correlation_to_start.insert(
      std::pair<uint32_t, std::chrono::time_point<std::chrono::system_clock>>(
          correlationId, t1));

  // this->tid_to_time.insert(
  // std::pair<uint32_t, time_points_t>(cbInfo->correlationId, time_point));
  this->correlation_to_function.insert(std::pair<uint32_t, const char *>(
      cbInfo->correlationId, cbInfo->functionName));
  this->correlation_to_symbol.insert(std::pair<uint32_t, const char *>(
      cbInfo->correlationId, cbInfo->symbolName));
  this->cid_to_tid.insert(std::pair<uint32_t, uint32_t>(
      correlationId, cprof::model::get_thread_id()));
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

void KernelCallTime::flush_tracers() {
  if (Profiler::instance().manager_->enable_zipkin()) {
    Profiler::instance().manager_->memcpy_tracer->Close();
    Profiler::instance().manager_->launch_tracer->Close();
  }
}
