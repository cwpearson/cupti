#include <string>

#include "cupti_activity_tracing.hpp"

//
// KERNEL
//
void handleKernel(const CUpti_ActivityKernel3 *record) {
  assert(record);
  const char *name = record->name;

  const uint32_t correlationId = record->correlationId;

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(record->start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(record->end);
  auto end_time_stamp =
      std::chrono::duration_cast<std::chrono::microseconds>(end_dur);

  Profiler::instance().chrome_tracer().complete_event(
      name, {}, start_time_point.count(),
      end_time_stamp.count() - start_time_point.count(), "kernel", "tid");
}

//
// MEMCPY
//
void handleMemcpy(const CUpti_ActivityMemcpy *record) {
  assert(record);
  const uint32_t correlationId = record->correlationId;

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(record->start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(record->end);
  auto end_time_stamp =
      std::chrono::duration_cast<std::chrono::microseconds>(end_dur);

  Profiler::instance().chrome_tracer().complete_event(
      std::to_string(record->bytes), {}, start_time_point.count(),
      end_time_stamp.count() - start_time_point.count(), "memcpy", "tid");
}

void tracing_activityHander(const CUpti_Activity *record) {

  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_KERNEL: {
    auto activity_cast =
        reinterpret_cast<const CUpti_ActivityKernel3 *>(record);
    handleKernel(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    auto activity_cast = reinterpret_cast<const CUpti_ActivityMemcpy *>(record);
    handleMemcpy(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_ENVIRONMENT: {
    auto activity_cast = (CUpti_ActivityEnvironment *)record;
    // addEnvironmentActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_OVERHEAD: {
    auto activity_cast = (CUpti_ActivityOverhead *)record;
    // addOverheadActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_GLOBAL_ACCESS: {
    auto activity_cast = (CUpti_ActivityGlobalAccess2 *)record;
    // addGlobalAccessActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_CUDA_EVENT: {
    auto activity_cast = (CUpti_ActivityCudaEvent *)record;
    // addCudaEventActivityAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_DRIVER: {
    auto activity_cast = (CUpti_ActivityAPI *)record;
    // addCudaDriverAndRuntimeAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    auto activity_cast = (CUpti_ActivityAPI *)record;
    // addCudaDriverAndRuntimeAnnotations(activity_cast);
    break;
  }
  case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
    auto activity_cast = (CUpti_ActivitySynchronization *)record;
    // addCudaActivitySynchronizationAnnotations(activity_cast);
    break;
  }
  default: {
    //   auto activity_cast = record;
  }
  };
}