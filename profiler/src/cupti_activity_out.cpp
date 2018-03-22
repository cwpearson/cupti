#include <string>

#include "cprof/activity/compute.hpp"
#include "cprof/activity/transfer.hpp"

#include "cupti_activity_out.hpp"
#include "profiler.hpp"

//
// KERNEL
//
static void handleKernel(const CUpti_ActivityKernel3 *record) {
  assert(record);
  auto compute = cprof::activity::Compute(record);
  profiler::atomic_out(compute.to_json_string() + "\n");
}

//
// MEMCPY
//
static void handleMemcpy(const CUpti_ActivityMemcpy *record) {
  assert(record);
  auto transfer = cprof::activity::Transfer(record);
  profiler::atomic_out(transfer.to_json_string() + "\n");
}

void out_activityHander(const CUpti_Activity *record) {

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
  default: {
    //   auto activity_cast = record;
    break;
  }
  };
}