#define __STDC_FORMAT_MACROS

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
#include <zipkin/span_context.h>

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
std::map<uintptr_t, zipkin::SpanContext> KernelCallTime::ptr_to_span;


static ZipkinOtTracerOptions options;
static ZipkinOtTracerOptions memcpy_tracer_options;
static ZipkinOtTracerOptions launch_tracer_options;
static std::shared_ptr<opentracing::Tracer> tracer;
static std::shared_ptr<opentracing::Tracer> memcpy_tracer;
static std::shared_ptr<opentracing::Tracer> launch_tracer;
static span_t parent_span;

KernelCallTime &KernelCallTime::instance() {
  options.service_name = "Parent";
  memcpy_tracer_options.service_name = "Memory Copy";
  launch_tracer_options.service_name = "Kernel Launch";
  if (!tracer) {
    // opentracing::Tracer::InitGlobal(tracer);
    tracer = makeZipkinOtTracer(options);
    memcpy_tracer = makeZipkinOtTracer(memcpy_tracer_options);
    launch_tracer = makeZipkinOtTracer(launch_tracer_options);
    
    parent_span = tracer->StartSpan("Parent");
  }
  static KernelCallTime a;
  return a;
}

std::shared_ptr<char> cppDemangle(const char *abiName) {
  int status;
  char *ret = abi::__cxa_demangle(abiName, 0, 0, &status);

  /* NOTE: must free() the returned char when done with it! */
  std::shared_ptr<char> retval;
  retval.reset((char *)ret, [](char *mem) {
    if (mem)
      free((void *)mem);
  });
  return retval;
}

KernelCallTime::KernelCallTime() {}

void KernelCallTime::kernel_start_time(const CUpti_CallbackData *cbInfo) {

  char* cudaMem = "cudaMemcpy";

  auto correlationId = cbInfo->correlationId;
  uint64_t startTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &startTimeStamp);
  time_points_t time_point;
  time_point.start_time = startTimeStamp;
  const char *memcpy = "cudaMemcpy";

  if (strcmp(cbInfo->functionName, memcpy) == 0) {
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    memcpy_info_t memInfo;
    std::cout << "Memcpy id: " << correlationId << std::endl;
    std::cout << "Memcpy size: " << params->count << std::endl;
    std::cout << "Source: " << (uintptr_t *) params->src << std::endl;
    std::cout << "Destination: " << (uintptr_t *) params->dst << std::endl;

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

  this->tid_to_time.insert(
      std::pair<uint32_t, time_points_t>(cbInfo->correlationId, time_point));
  this->correlation_to_function.insert(std::pair<uint32_t, const char *>(
      cbInfo->correlationId, cbInfo->functionName));
  this->correlation_to_symbol.insert(std::pair<uint32_t, const char *>(
      cbInfo->correlationId, cbInfo->symbolName));
}

void KernelCallTime::kernel_end_time(const CUpti_CallbackData *cbInfo) {
  cudaDeviceSynchronize();
  auto correlationId = cbInfo->correlationId;
  auto t2 = SystemClock::now();
  auto t1 = this->correlation_to_start.find(correlationId)->second;
  span_t current_span;
  const char *launchVal = "cudaLaunch";

  ZipkinOtTracerOptions options;
  options.service_name = "Kernel Launch " + std::to_string(correlationId);
  std::shared_ptr<opentracing::Tracer> temp_launch_tracer = makeZipkinOtTracer(options);
  
  if (strcmp(cbInfo->functionName, launchVal) == 0){
    
        for (auto iter = this->correlation_to_dest.begin(); iter != this->correlation_to_dest.end(); iter++) {
          for (size_t argIdx = 0; argIdx < ConfiguredCall().args.size(); ++argIdx) { 
            if (ConfiguredCall().args[argIdx] == iter->second){
              //Dependency exists!
              auto spanIter = this->ptr_to_span.find(ConfiguredCall().args[argIdx]);
              if (spanIter != this->ptr_to_span.end()){
                auto memContext = spanIter->second;
                printf("Before segfault\n");
                current_span = temp_launch_tracer->StartSpan(std::to_string(correlationId),
                {ChildOf(&memContext)});
                temp_launch_tracer->Close();
                printf("After segfault\n");
              }
  
              break;
            }
       
          }
        }
  } else {
    current_span = memcpy_tracer->StartSpan(std::to_string(correlationId),
                      {ChildOf(&parent_span->context())});

    auto iter = this->correlation_to_dest.find(correlationId);
    zipkin::SpanContext tempContext();
    this->ptr_to_span.insert(std::pair<uintptr_t, const zipkin::SpanContext>(iter->second, tempContext));

  }
   
  current_span->SetTag(
      "Function Name",
      this->correlation_to_function.find(correlationId)->second);
  auto cStrSymbol = this->correlation_to_symbol.find(correlationId)->second;
  if (cStrSymbol != NULL) {
    current_span->SetTag("Symbol ", cStrSymbol);
  }

  if (this->correlation_id_to_info.find(correlationId) !=
      this->correlation_id_to_info.end()) {
    auto memCpyIter = this->correlation_id_to_info.find(correlationId);
    auto memCpyInfo = memCpyIter->second;
    current_span->SetTag("Transfer size", memCpyInfo.memcpySize);
    current_span->SetTag("Transfer type",
                         memcpy_type_to_string(memCpyInfo.memcpyType));
  }

  current_span->Finish();
  uint64_t endTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
  auto time_point_iterator = this->tid_to_time.find(cbInfo->correlationId);
  time_point_iterator->second.end_time = endTimeStamp;

}

char *KernelCallTime::memcpy_type_to_string(cudaMemcpyKind kind) {
  switch (kind) {
  case cudaMemcpyHostToHost: {
    static char *HtH = "cudaMemcpyHostToHost";
    return HtH;
  }
  case cudaMemcpyHostToDevice: {
    static char *HtD = "cudaMemcpyHostToDevice";
    return HtD;
  }
  case cudaMemcpyDeviceToHost: {
    static char *DtH = "cudaMemcpyDeviceToHost";
    return DtH;
  }
  case cudaMemcpyDeviceToDevice: {
    static char *DtD = "cudaMemcpyDeviceToDevice";
    return DtD;
  }
  case cudaMemcpyDefault: {
    static char *defaultDir = "cudaMemcpyDefault";
    return defaultDir;
  }
  default: {
    static char *unknown = "Unknown";
    return "Unknown";
  }
  }
}

void KernelCallTime::write_to_file() {
  parent_span->Finish();
  tracer->Close();
  memcpy_tracer->Close();
  launch_tracer->Close();
}