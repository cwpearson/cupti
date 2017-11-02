#define __STDC_FORMAT_MACROS

#include <inttypes.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <inttypes.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cxxabi.h> 


#include "kernel_time.hpp"

using namespace std::chrono;
using namespace zipkin;
using namespace opentracing;

std::map<uint32_t, time_points_t> KernelCallTime::tid_to_time;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_function;
std::map<uint32_t, const char *> KernelCallTime::correlation_to_symbol;
std::map<uint32_t, std::chrono::time_point<std::chrono::system_clock> > KernelCallTime::correlation_to_start;
std::map<uint32_t, memcpy_info_t> KernelCallTime::correlation_id_to_info;

static ZipkinOtTracerOptions options;
static std::shared_ptr<opentracing::Tracer> tracer;
static span_t parent_span;



KernelCallTime &KernelCallTime::instance() {
  options.service_name = "CUPTI";
  if (!tracer){
    tracer = makeZipkinOtTracer(options);
    parent_span = tracer->StartSpan("Parent");
  }
  static KernelCallTime a;
  return a;
}

std::shared_ptr<char> cppDemangle(const char *abiName)
{
  int status;    
  char *ret = abi::__cxa_demangle(abiName, 0, 0, &status);  

  /* NOTE: must free() the returned char when done with it! */
  std::shared_ptr<char> retval;
  retval.reset( (char *)ret, [](char *mem) { if (mem) free((void*)mem); } );
  return retval;
}

KernelCallTime::KernelCallTime() {}

void KernelCallTime::kernel_start_time(const CUpti_CallbackData *cbInfo) {
  auto correlationId = cbInfo->correlationId;
  uint64_t startTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &startTimeStamp);
  time_points_t time_point;
  time_point.start_time = startTimeStamp;
  // //std::cout << cbInfo->correlationId << " start time stamp " << startTimeStamp << std::endl;
  const char * memcpy = "cudaMemcpy";

  if (strcmp(cbInfo->functionName, memcpy) == 0){
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    memcpy_info_t memInfo;
    std::cout << "Memcpy id: " << correlationId << std::endl;
    std::cout << "Memcpy size: " << params->count << std::endl;
    memInfo.memcpyType = params->kind;
    memInfo.memcpySize = params->count;
    this->correlation_id_to_info.insert(std::pair<uint32_t, memcpy_info_t>(correlationId, memInfo));
  } 

  auto t1 = SystemClock::now();
  this->correlation_to_start.insert(std::pair<uint32_t, std::chrono::time_point<std::chrono::system_clock> >(correlationId, t1));
  
  this->tid_to_time.insert(
      std::pair<uint32_t, time_points_t>(cbInfo->correlationId, time_point));
  this->correlation_to_function.insert(std::pair<uint32_t, const char *>(
      cbInfo->correlationId, cbInfo->functionName));
  this->correlation_to_symbol.insert(std::pair<uint32_t, const char*>(cbInfo->correlationId, cbInfo->symbolName));
}

void KernelCallTime::kernel_end_time(const CUpti_CallbackData *cbInfo) {
  cudaDeviceSynchronize();
  auto correlationId = cbInfo->correlationId;  
  auto t2 = SystemClock::now();
  auto t1 = this->correlation_to_start.find(correlationId)->second;
  span_t current_span = tracer->StartSpan(std::to_string(correlationId), {ChildOf(&parent_span->context()), StartTimestamp(t1)});
  current_span->SetTag("Function Name", this->correlation_to_function.find(correlationId)->second);  
  auto cStrSymbol = this->correlation_to_symbol.find(correlationId)->second;
  if (cStrSymbol != NULL){
    current_span->SetTag("Symbol ", cStrSymbol);  
  }

  if (this->correlation_id_to_info.find(correlationId) != this->correlation_id_to_info.end()){
    auto memCpyIter = this->correlation_id_to_info.find(correlationId);
    auto memCpyInfo = memCpyIter->second;
    current_span->SetTag("Transfer size", memCpyInfo.memcpySize);
    current_span->SetTag("Transfer type", memcpy_type_to_string(memCpyInfo.memcpyType));
  }
  
  current_span->Finish();
  uint64_t endTimeStamp;
  cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
  auto time_point_iterator = this->tid_to_time.find(cbInfo->correlationId);
  time_point_iterator->second.end_time = endTimeStamp;
  // //std::cout << cbInfo->correlationId << " End time stamp " << endTimeStamp << std::endl;
}

char* KernelCallTime::memcpy_type_to_string(cudaMemcpyKind kind){
  switch (kind){
    case cudaMemcpyHostToHost:
    {
      static char *HtH= "cudaMemcpyHostToHost";
      return HtH;
    }
    case cudaMemcpyHostToDevice:
    {
      static char *HtD = "cudaMemcpyHostToDevice";
      return HtD;
    }
    case cudaMemcpyDeviceToHost:
    {
      static char *DtH = "cudaMemcpyDeviceToHost";
      return DtH;
    }
    case cudaMemcpyDeviceToDevice:
    {
      static char *DtD = "cudaMemcpyDeviceToDevice";
      return DtD;
    }
    case cudaMemcpyDefault:
    {
      static char *defaultDir = "cudaMemcpyDefault";
      return defaultDir;
    }
    default:
    {
      static char *unknown = "Unknown";
      return "Unknown";
    }
  }
}


void KernelCallTime::write_to_file() {
  parent_span->Finish();
  tracer->Close();
  // printf("KernelCallTime write to file\n");
  using boost::property_tree::ptree;
  using boost::property_tree::write_json;
  ptree pt;
  
  long long tempTime;
  for (auto iter = this->tid_to_time.begin(); iter != this->tid_to_time.end();
       iter++) {
    tempTime =  iter->second.end_time - iter->second.start_time;
    pt.put("correlationId", std::to_string(iter->first));

    auto cStrSymbol = this->correlation_to_symbol.find(iter->first)->second;
    //Symbol name is only valid for driver and runtime launch callbacks
    if (cStrSymbol != NULL){
      std::string s(cStrSymbol);
      pt.put("symbol", s);
    }

    pt.put("functionName", this->correlation_to_function.find(iter->first)->second);
    pt.put("startTime", std::to_string(iter->second.start_time));
    pt.put("endTime", std::to_string(iter->second.end_time));
    pt.put("timeSpan", std::to_string(tempTime));
    // write_json(//std::cout, pt);  
    pt.clear();
  }
}