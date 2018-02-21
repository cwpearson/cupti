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
// Libraries required to write to JSON
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>

#include "cupti_subscriber.hpp"
#include "profiler.hpp"
#include "timer.hpp"

using namespace std::chrono;
using namespace zipkin;
using namespace opentracing;

void Timer::callback_add_annotations(const CUpti_CallbackData *cbInfo,
                                     CUpti_CallbackId cbid) {

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;

  std::string functionName(cbInfo->functionName, strlen(cbInfo->functionName));
  std::string symbolName(cbInfo->symbolName, strlen(cbInfo->symbolName));

  if (Profiler::instance().is_zipkin_enabled()) {
    span_t current_span;
    current_span = Profiler::instance().manager_->launch_tracer->StartSpan(
        std::to_string(cbInfo->correlationId),
        {FollowsFrom(&Profiler::instance().manager_->parent_span->context())});
    current_span->SetTag("contextUid", std::to_string(cbInfo->contextUid));
    current_span->SetTag("functionName", functionName);
    if (cbInfo->symbolName != NULL) {
      current_span->SetTag("symbolName", symbolName);
    }
    current_span->Finish();
  }

  if (cbInfo->symbolName != NULL) {
    pt.put("symbolName", symbolName);
  }

  // Fill in to parse the different arguments depending on what type of driver
  // call
  switch (cbid) {
  case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
    break;
  default:
    profiler::err() << "Unsupported cbid, cannot get arguments" << std::endl;
    break;
  }

  pt.put("contextUid", std::to_string(cbInfo->contextUid));
  pt.put("functionName", functionName);
  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());

  // To fill in with various data
}

/*CUDA8*/
void Timer::addKernelActivityAnnotations(
    CUpti_ActivityKernel3 *kernel_Activity) {

  CUpti_ActivityKernel3 local_Kernel_Activity = *kernel_Activity;

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(local_Kernel_Activity.start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(local_Kernel_Activity.end);
  auto end_time_stamp =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);

  if (Profiler::instance().is_zipkin_enabled()) {
    span_t current_span;
    current_span = Profiler::instance().manager_->launch_tracer->StartSpan(
        std::to_string(local_Kernel_Activity.correlationId),
        {FollowsFrom(&Profiler::instance().manager_->parent_span->context()),
         StartTimestamp(start_time_point)});
    // Extract useful information from local_Kernel_Activity and add it to trace
    current_span->SetTag("blockX",
                         std::to_string(local_Kernel_Activity.blockX));
    current_span->SetTag("blockY",
                         std::to_string(local_Kernel_Activity.blockY));
    current_span->SetTag("blockZ",
                         std::to_string(local_Kernel_Activity.blockZ));
    current_span->SetTag("completed",
                         std::to_string(local_Kernel_Activity.completed));
    current_span->SetTag("deviceId",
                         std::to_string(local_Kernel_Activity.deviceId));
    current_span->SetTag(
        "dynamicSharedMemory",
        std::to_string(local_Kernel_Activity.dynamicSharedMemory));
    current_span->SetTag("gridId",
                         std::to_string(local_Kernel_Activity.gridId));
    current_span->SetTag("gridX", std::to_string(local_Kernel_Activity.gridX));
    current_span->SetTag("gridY", std::to_string(local_Kernel_Activity.gridY));
    current_span->SetTag("gridZ", std::to_string(local_Kernel_Activity.gridZ));
    current_span->SetTag(
        "localMemoryPerThread",
        std::to_string(local_Kernel_Activity.localMemoryPerThread));
    current_span->SetTag(
        "localMemoryTotal",
        std::to_string(local_Kernel_Activity.localMemoryTotal));
    current_span->SetTag(
        "registersPerThread",
        std::to_string(local_Kernel_Activity.registersPerThread));
    current_span->SetTag(
        "sharedMemoryConfig",
        std::to_string(local_Kernel_Activity.sharedMemoryConfig));
    current_span->SetTag(
        "staticSharedMemory",
        std::to_string(local_Kernel_Activity.staticSharedMemory));
    current_span->SetTag("streamId",
                         std::to_string(local_Kernel_Activity.streamId));
    current_span->SetTag(
        "staticSharedMemory",
        std::to_string(local_Kernel_Activity.staticSharedMemory));
    current_span->Finish({FinishTimestamp(end_time_stamp)});
  }

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;

  pt.put("correlation_id", std::to_string(local_Kernel_Activity.correlationId));
  pt.put("blockX", std::to_string(local_Kernel_Activity.blockX));
  pt.put("blockY", std::to_string(local_Kernel_Activity.blockY));
  pt.put("blockZ", std::to_string(local_Kernel_Activity.blockZ));
  pt.put("completed", std::to_string(local_Kernel_Activity.completed));
  pt.put("deviceId", std::to_string(local_Kernel_Activity.deviceId));
  pt.put("dynamicSharedMemory",
         std::to_string(local_Kernel_Activity.dynamicSharedMemory));
  pt.put("gridId", std::to_string(local_Kernel_Activity.gridId));
  pt.put("gridX", std::to_string(local_Kernel_Activity.gridX));
  pt.put("gridY", std::to_string(local_Kernel_Activity.gridY));
  pt.put("gridZ", std::to_string(local_Kernel_Activity.gridZ));
  pt.put("localMemoryPerThread",
         std::to_string(local_Kernel_Activity.localMemoryPerThread));
  pt.put("localMemoryTotal",
         std::to_string(local_Kernel_Activity.localMemoryTotal));
  pt.put("registersPerThread",
         std::to_string(local_Kernel_Activity.registersPerThread));
  pt.put("sharedMemoryConfig",
         std::to_string(local_Kernel_Activity.sharedMemoryConfig));
  pt.put("staticSharedMemory",
         std::to_string(local_Kernel_Activity.staticSharedMemory));
  pt.put("streamId", std::to_string(local_Kernel_Activity.streamId));
  pt.put("staticSharedMemory",
         std::to_string(local_Kernel_Activity.staticSharedMemory));
  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

void Timer::addMemcpyActivityAnnotations(
    CUpti_ActivityMemcpy *memcpy_Activity) {
  CUpti_ActivityMemcpy local_Memcpy_Activity = *memcpy_Activity;

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(local_Memcpy_Activity.start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(local_Memcpy_Activity.end);
  auto end_time_stamp =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_dur);
  if (Profiler::instance().is_zipkin_enabled()) {
    span_t current_span;
    current_span = Profiler::instance().manager_->memcpy_tracer->StartSpan(
        std::to_string(local_Memcpy_Activity.correlationId),
        {FollowsFrom(&Profiler::instance().manager_->parent_span->context()),
         StartTimestamp(start_time_point)});
    current_span->SetTag("bytes", std::to_string(local_Memcpy_Activity.bytes));
    current_span->SetTag("contextId",
                         std::to_string(local_Memcpy_Activity.contextId));
    current_span->SetTag("copyKind",
                         std::to_string(local_Memcpy_Activity.copyKind));
    current_span->SetTag("deviceId",
                         std::to_string(local_Memcpy_Activity.deviceId));
    current_span->SetTag("dstKind",
                         std::to_string(local_Memcpy_Activity.dstKind));
    current_span->SetTag("flags", std::to_string(local_Memcpy_Activity.flags));
    current_span->SetTag(
        "runtimeCorrelationId",
        std::to_string(local_Memcpy_Activity.runtimeCorrelationId));
    current_span->SetTag("srcKind",
                         std::to_string(local_Memcpy_Activity.srcKind));
    current_span->SetTag("streamId",
                         std::to_string(local_Memcpy_Activity.streamId));
    current_span->Finish({FinishTimestamp(end_time_stamp)});
  }

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;
  pt.put("bytes", std::to_string(local_Memcpy_Activity.bytes));
  pt.put("contextId", std::to_string(local_Memcpy_Activity.contextId));
  pt.put("copyKind", std::to_string(local_Memcpy_Activity.copyKind));
  pt.put("deviceId", std::to_string(local_Memcpy_Activity.deviceId));
  pt.put("dstKind", std::to_string(local_Memcpy_Activity.dstKind));
  pt.put("flags", std::to_string(local_Memcpy_Activity.flags));
  pt.put("runtimeCorrelationId",
         std::to_string(local_Memcpy_Activity.runtimeCorrelationId));
  pt.put("srcKind", std::to_string(local_Memcpy_Activity.srcKind));
  pt.put("streamId", std::to_string(local_Memcpy_Activity.streamId));
  std::ostringstream buf;
  write_json(buf, pt, false);
}

void Timer::addEnvironmentActivityAnnotations(
    CUpti_ActivityEnvironment *environment_Activity) {
  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;
  switch (environment_Activity->environmentKind) {
  case CUPTI_ACTIVITY_ENVIRONMENT_POWER: {
    auto powerAct = environment_Activity;
    pt.put("activityType", std::to_string(CUPTI_ACTIVITY_ENVIRONMENT_POWER));
    pt.put("power", std::to_string(powerAct->data.power.power));
    pt.put("deviceId", std::to_string(powerAct->deviceId));
    pt.put("timestamp", std::to_string(powerAct->timestamp));

    std::ostringstream buf;
    write_json(buf, pt, false);
    logging::atomic_out(buf.str());
    return;
  };
  default: {
    profiler::err() << "Environment data type not supported "
                    << environment_Activity->environmentKind << std::endl;
    return;
  }
  }
}

void Timer::addOverheadActivityAnnotations(
    CUpti_ActivityOverhead *overhead_Activity) {
  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;

  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

void Timer::addGlobalAccessActivityAnnotations(
    CUpti_ActivityGlobalAccess2 *overhead_GlobalAccess) {
  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;

  pt.put("correlationId", std::to_string(overhead_GlobalAccess->correlationId));
  pt.put("functionName", std::to_string(overhead_GlobalAccess->functionId));
  pt.put("executed", std::to_string(overhead_GlobalAccess->executed));
  pt.put("threadsExecuted",
         std::to_string(overhead_GlobalAccess->threadsExecuted));

  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

/**
 *
 * */
void Timer::addCudaEventActivityAnnotations(
    CUpti_ActivityCudaEvent *event_Activity) {
  /**
   * CUpti_ActivityCudaEvent
    uint32_t  contextId
    uint32_t  correlationId
    uint32_t  eventId
    CUpti_ActivityKind kind
    uint32_t  pad
    uint32_t  streamId
   * Read more at: http://docs.nvidia.com/cuda/cupti/index.html#ixzz57QBm8lWC
   * */

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;
  pt.put("contextId", std::to_string(event_Activity->contextId));
  pt.put("correlationId", std::to_string(event_Activity->correlationId));
  pt.put("eventId", std::to_string(event_Activity->correlationId));
  pt.put("pad", std::to_string(event_Activity->pad));
  pt.put("streamId", std::to_string(event_Activity->streamId));

  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

void Timer::addCudaDriverAndRuntimeAnnotations(
    CUpti_ActivityAPI *activity_Activity) {
  /**
   * CUpti_ActivityAPI
   *CUpti_CallbackId cbid
    uint32_t  correlationId
    uint64_t  end
    CUpti_ActivityKind kind
    uint32_t  processId
    uint32_t  returnValue
    uint64_t  start
    uint32_t  threadId


    Read more at: http://docs.nvidia.com/cuda/cupti/index.html#ixzz57RpPGxZA
   * */
  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;
  pt.put("correlationId", std::to_string(activity_Activity->correlationId));
  pt.put("end", std::to_string(activity_Activity->end));
  pt.put("processId", std::to_string(activity_Activity->processId));
  pt.put("start", std::to_string(activity_Activity->start));
  pt.put("threadId", std::to_string(activity_Activity->threadId));

  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

void Timer::addCudaActivitySynchronizationAnnotations(
    CUpti_ActivitySynchronization *synchronization_Activity) {
  /**
   * CUpti_ActivitySynchronization
    uint32_t  contextId
    uint32_t  correlationId
    uint32_t  cudaEventId
    uint64_t  end
    CUpti_ActivityKind kind
    uint64_t  start
    uint32_t  streamId
    CUpti_ActivitySynchronizationType type
    Read more at: http://docs.nvidia.com/cuda/cupti/index.html#ixzz57RnfhfUV
   * */

  using boost::property_tree::ptree;
  using boost::property_tree::write_json;

  ptree pt;
  pt.put("contextId", std::to_string(synchronization_Activity->contextId));
  pt.put("correlationId",
         std::to_string(synchronization_Activity->correlationId));
  pt.put("cudaEventId", std::to_string(synchronization_Activity->cudaEventId));
  pt.put("end", std::to_string(synchronization_Activity->end));
  pt.put("start", std::to_string(synchronization_Activity->start));
  pt.put("streamId", std::to_string(synchronization_Activity->streamId));
  pt.put("streamId", std::to_string(synchronization_Activity->streamId));

  std::ostringstream buf;
  write_json(buf, pt, false);
  logging::atomic_out(buf.str());
}

void Timer::activity_add_annotations(CUpti_Activity *activity_data) {
  switch (activity_data->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL: {
    auto activity_cast = (CUpti_ActivityKernel3 *)activity_data;
    addKernelActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    auto activity_cast = (CUpti_ActivityMemcpy *)activity_data;
    addMemcpyActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_ENVIRONMENT: {
    auto activity_cast = (CUpti_ActivityEnvironment *)activity_data;
    addEnvironmentActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_OVERHEAD: {
    auto activity_cast = (CUpti_ActivityOverhead *)activity_data;
    addOverheadActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS: {
    auto activity_cast = (CUpti_ActivityGlobalAccess2 *)activity_data;
    addGlobalAccessActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_CUDA_EVENT: {
    auto activity_cast = (CUpti_ActivityCudaEvent *)activity_data;
    addCudaEventActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_DRIVER: {
    auto activity_cast = (CUpti_ActivityAPI *)activity_data;
    addCudaDriverAndRuntimeAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    auto activity_cast = (CUpti_ActivityAPI *)activity_data;
    addCudaDriverAndRuntimeAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
    auto activity_cast = (CUpti_ActivitySynchronization *)activity_data;
    addCudaActivitySynchronizationAnnotations(activity_cast);
    break;
  }
  default: { auto activity_cast = activity_data; }
  };
}