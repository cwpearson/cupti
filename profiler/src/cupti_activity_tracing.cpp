#include <string>

#include "cprof/activity/compute.hpp"
#include "cprof/activity/transfer.hpp"
#include "cprof/chrome_tracing/complete_event.hpp"

#include "cupti_activity_tracing.hpp"

//
// KERNEL
//
void handleKernel(const CUpti_ActivityKernel3 *record) {
  assert(record);
  auto compute = cprof::activity::Compute(record);
  Profiler::instance().chrome_tracer().write_event(
      compute.chrome_complete_event());
  profiler::atomic_out(compute.json());
}

//
// MEMCPY
//
void handleMemcpy(const CUpti_ActivityMemcpy *record) {
  assert(record);
  auto transfer = cprof::activity::Transfer(record);
  Profiler::instance().chrome_tracer().write_event(
      transfer.chrome_complete_event());
  profiler::atomic_out(transfer.json());
}

//
// Overhead
//
void handleOverhead(const CUpti_ActivityOverhead *record) {
  assert(record);

  /*Get start and end times for kernel*/
  std::chrono::nanoseconds start_dur(record->start);
  auto start_time_point =
      std::chrono::duration_cast<std::chrono::microseconds>(start_dur);
  std::chrono::nanoseconds end_dur(record->end);
  auto end_time_stamp =
      std::chrono::duration_cast<std::chrono::microseconds>(end_dur);

  auto event = cprof::chrome_tracing::CompleteEventUs(
      "", {}, start_time_point.count(),
      end_time_stamp.count() - start_time_point.count(), "profiler",
      "cupti overhead");

  Profiler::instance().chrome_tracer().write_event(event);
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
    auto activity_cast =
        reinterpret_cast<const CUpti_ActivityOverhead *>(record);
    handleOverhead(activity_cast);
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