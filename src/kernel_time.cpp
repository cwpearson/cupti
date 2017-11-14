#include <inttypes.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <chrono>
#include <ctime>
#include <cxxabi.h>
#include <inttypes.h>
#include <iomanip>
#include <iostream>

#include "callbacks.hpp"
#include "kernel_time.hpp"

using namespace std::chrono;
using namespace zipkin;
using namespace opentracing;

std::map<uint32_t, time_points_t> KernelCallTime::tid_to_time;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_function;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_symbol;
std::map<uint32_t, std::chrono::time_point<std::chrono::system_clock>>
    KernelCallTime::correlation_to_start;
std::map<uint32_t, memcpy_info_t> KernelCallTime::correlation_id_to_info;
std::map<uint32_t, uintptr_t> KernelCallTime::correlation_to_dest;
std::map<uintptr_t, TextMapCarrier> KernelCallTime::ptr_to_span;
std::unordered_map<std::string, std::string> KernelCallTime::text_map;
std::map<uint32_t, std::vector<uintptr_t> > KernelCallTime::cid_to_call;



static ZipkinOtTracerOptions options;
static ZipkinOtTracerOptions memcpy_tracer_options;
static ZipkinOtTracerOptions launch_tracer_options;
static std::shared_ptr<opentracing::Tracer> tracer;
static std::shared_ptr<opentracing::Tracer> memcpy_tracer;
static std::shared_ptr<opentracing::Tracer> launch_tracer;
static span_t parent_span;

static std::unordered_map<std::string, std::string> text_map;
static TextMapCarrier carrier(text_map);

KernelCallTime &KernelCallTime::instance() {
  
  options.service_name = "Parent";
  memcpy_tracer_options.service_name = "Memory Copy";
  launch_tracer_options.service_name = "Kernel Launch";
  if (!tracer) {
    tracer = makeZipkinOtTracer(options);
    memcpy_tracer = makeZipkinOtTracer(memcpy_tracer_options);
    launch_tracer = makeZipkinOtTracer(launch_tracer_options);
    
    parent_span = tracer->StartSpan("Parent");
  }
  static KernelCallTime a;
  return a;
}

KernelCallTime::KernelCallTime() {}

void KernelCallTime::kernel_start_time(const CUpti_CallbackData *cbInfo) {
  uint64_t startTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &startTimeStamp);
  char* cudaMem = "cudaMemcpy";
  auto correlationId = cbInfo->correlationId;
  time_points_t time_point;
  time_point.start_time = startTimeStamp;
  const char *memcpy = "cudaMemcpy";

  if (strcmp(cbInfo->functionName, memcpy) == 0) {
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    memcpy_info_t memInfo;

    this->correlation_to_dest.insert(std::pair<uint32_t, uintptr_t>(correlationId, (uintptr_t)params->dst));
    
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
}

void KernelCallTime::kernel_end_time(const CUpti_CallbackData *cbInfo) {
  // uint64_t endTimeStamp;
  // cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
  // auto correlationId = cbInfo->correlationId;
  // auto t2 = SystemClock::now();
  // auto t1 = this->correlation_to_start.find(correlationId)->second;
  // auto milliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  // auto ms = milliseconds.count();
  // // std::cout << ms << "ms" << std::endl;

  // span_t current_span;
  // const char *launchVal = "cudaLaunch";

  // ZipkinOtTracerOptions options;
  // options.service_name = "Kernel Launch " + std::to_string(correlationId);

  // bool found = false;

  // if (strcmp(cbInfo->functionName, launchVal) == 0){
  //   return;
  //   // for (auto iter = this->correlation_to_dest.begin(); iter != this->correlation_to_dest.end(); iter++) {
  //   //   for (size_t argIdx = 0; argIdx < ConfiguredCall().args.size(); ++argIdx) { 
  //   //     if (ConfiguredCall().args[argIdx] == iter->second){
  //   //       // Dependency exists!
  //   //       auto spanIter = this->ptr_to_span.find(ConfiguredCall().args[argIdx]);
  //   //       if (spanIter != this->ptr_to_span.end()){ 
                
  //   //         auto span_context_maybe = launch_tracer->Extract(carrier);
  //   //         assert(span_context_maybe);
  //   //         auto contextPtr = spanIter->second;     
  //   //         if (!found){
  //   //           current_span = launch_tracer->StartSpan(std::to_string(correlationId),
  //   //           {FollowsFrom(span_context_maybe->get()), StartTimestamp(t1)});  
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
  // //                     {ChildOf(&parent_span->context()), StartTimestamp(t1)});

  //   auto iter = this->correlation_to_dest.find(correlationId);
   

  //   auto err = tracer->Inject(current_span->context(), carrier);
  //   assert(err);

    // this->ptr_to_cid.insert(std::pair<uintptr_t, TextMapCarrier>(iter->second, carrier));
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

    
  //   auto cStrSymbol = this->correlation_to_symbol.find(correlationId)->second;
  //   if (cStrSymbol != NULL) {
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
  //   std::cout <<  "Function Name " << cbInfo->functionName << " Time from cuda " << time_point.end_time - time_point.start_time << std::endl;
}

const char *KernelCallTime::memcpy_type_to_string(uint8_t kind) {
  switch (kind) {
  case cudaMemcpyHostToHost: {
    static const char *HtH = "cudaMemcpyHostToHost";
    return HtH;
  }
  case cudaMemcpyHostToDevice: {
    static const char *HtD = "cudaMemcpyHostToDevice";
    return HtD;
  }
  case cudaMemcpyDeviceToHost: {
    static const char *DtH = "cudaMemcpyDeviceToHost";
    return DtH;
  }
  case cudaMemcpyDeviceToDevice: {
    static const char *DtD = "cudaMemcpyDeviceToDevice";
    return DtD;
  }
  case cudaMemcpyDefault: {
    static const char *defaultDir = "cudaMemcpyDefault";
    return defaultDir;
  }
  default: {
    static const char *unknown = "Unknown";
    return "Unknown";
  }
  }
}

void KernelCallTime::memcpy_activity_times(CUpti_ActivityMemcpy * memcpyRecord){
  span_t current_span;
  auto correlationId = memcpyRecord->correlationId;

  std::chrono::nanoseconds start_dur(memcpyRecord->start);
  std::chrono::nanoseconds end_dur(memcpyRecord->end);

  std::cout << "Start pre truncate " << memcpyRecord->start << std::endl;
  std::cout << "End pre truncate " << memcpyRecord->end << std::endl;
  
  std::cout << "Start: "<< start_dur.count() << std::endl;
  std::cout << "End: " << end_dur.count() << std::endl;
  

  // std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start_dur{std::chrono::nanoseconds(memcpyRecord->start)};
  // std::chrono::time_point<std::chrono::steady_clock, std::chrono::nanoseconds> end_dur{std::chrono::nanoseconds(memcpyRecord->end)};
  
  
  // auto start_time_point = std::chrono::duration_cast<std::chrono::nanoseconds>(start_dur);
  // auto end_time_point = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);

  // std::chrono::time_point<std::chrono::system_clock> start_time_point(start_dur);
  // std::chrono::time_point<std::chrono::steady_clock> end_time_point(end_dur);

  current_span = memcpy_tracer->StartSpan(std::to_string(correlationId),
                      {ChildOf(&parent_span->context()), StartTimestamp(start_dur)});

  current_span->SetTag("Transfer size", memcpyRecord->bytes);
  current_span->SetTag("Transfer type",
                         memcpy_type_to_string(memcpyRecord->copyKind));
  
  auto err = tracer->Inject(current_span->context(), carrier);                       
  current_span->Finish({FinishTimestamp(end_dur)});   

  assert(err);

  auto iter = this->correlation_to_dest.find(correlationId);  
  this->ptr_to_span.insert(std::pair<uintptr_t, TextMapCarrier>(iter->second, carrier));
  memcpy_tracer->Close();
}

void KernelCallTime::kernel_activity_times(uint32_t cid, uint64_t startTime, uint64_t endTime,  CUpti_ActivityKernel3* launchRecord){

  auto found = false;
  span_t current_span;
  auto correlationId = cid;

  std::chrono::nanoseconds start_dur(startTime);
  std::chrono::nanoseconds end_dur(endTime);
  

  auto start_time_point = std::chrono::duration_cast<std::chrono::nanoseconds>(start_dur);
  auto end_time_point = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);
  auto configCallIter = this->cid_to_call.find(correlationId);
  // if (cid_to_call.end() != configCallIter){
  //   std::cout << "Here!" << std::endl;
  //   auto configCall = configCallIter->second;


  //   for (auto iter = this->correlation_to_dest.begin(); iter != this->correlation_to_dest.end(); iter++) {
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
  //                 current_span = launch_tracer->StartSpan(std::to_string(correlationId),
  //                   {FollowsFrom(span_context_maybe->get()), StartTimestamp(start_time_point)});  

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
   if (!found){
    current_span = launch_tracer->StartSpan(std::to_string(correlationId),
                   {FollowsFrom(&parent_span->context()), StartTimestamp(start_time_point)});  
  }

    current_span->SetTag(
      "Function Name",
      "cudaLaunch");
  


  current_span->SetTag("Current device", std::to_string(launchRecord->deviceId));

    
  //   auto cStrSymbol = this->correlation_to_symbol.find(correlationId)->second;
  //   if (cStrSymbol != NULL) {
      current_span->SetTag("Name ", launchRecord->name);
  //   }
 
  current_span->Finish({FinishTimestamp(end_time_point)});
  launch_tracer->Close();
}

void KernelCallTime::save_configured_call(uint32_t cid, std::vector<uintptr_t> configCall){
  this->cid_to_call.insert(std::pair<uint32_t, std::vector<uintptr_t>>(cid, configCall));
}

void KernelCallTime::close_parent(){
  parent_span->Finish();  
}

void KernelCallTime::write_to_file() {
  tracer->Close();
  memcpy_tracer->Close();
  launch_tracer->Close();
}