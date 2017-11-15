#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include "cprof/activity_callbacks.hpp"
#include "cprof/allocation_record.hpp"
#include "cprof/allocations.hpp"
#include "cprof/apis.hpp"
#include "cprof/backtrace.hpp"
#include "cprof/driver_state.hpp"
#include "cprof/hash.hpp"
#include "cprof/kernel_time.hpp"
#include "cprof/memory.hpp"
#include "cprof/memorycopykind.hpp"
#include "cprof/numa.hpp"
#include "cprof/thread.hpp"
#include "cprof/util_cuda.hpp"
#include "cprof/util_cupti.hpp"
#include "cprof/value.hpp"
#include "cprof/values.hpp"

// FIXME: this should be per-thread
typedef struct {
  dim3 gridDim;
  dim3 blockDim;
  size_t sharedMem;
  cudaStream_t stream;
  std::vector<uintptr_t> args;
  bool valid = false;
} ConfiguredCall_t;

ConfiguredCall_t &ConfiguredCall() {
  static ConfiguredCall_t cc;
  return cc;
}

static int counter = 0;

// Function that is called when a Kernel is called
// Record timing in this
static void handleCudaLaunch(Values &values, const CUpti_CallbackData *cbInfo) {
  printf("callback: cudaLaunch preamble\n");
  auto kernelTimer = KernelCallTime::instance();

  // print_backtrace();

  // Get the current stream
  const cudaStream_t stream = ConfiguredCall().stream;
  const char *symbolName = cbInfo->symbolName;
  printf("launching %s\n", symbolName);

  // Find all values that are used by arguments
  std::vector<Values::id_type> kernelArgIds;
  for (size_t argIdx = 0; argIdx < ConfiguredCall().args.size();
       ++argIdx) { // for each kernel argument
                   // printf("arg %lu, val %lu\n", argIdx, valIdx);

    // FIXME: assuming with p2p access, it could be on any device?
    const auto &kv =
        values.find_live_device(ConfiguredCall().args[argIdx], 1 /*size*/);

    const auto &key = kv.first;
    if (key != uintptr_t(nullptr)) {
      kernelArgIds.push_back(kv.first);
      printf("found val %lu for kernel arg=%lu\n", key,
             ConfiguredCall().args[argIdx]);
    }
  }

  if (kernelArgIds.empty()) {
    printf("WARN: didn't find any values for cudaLaunch\n");
  }
  // static std::map<Value::id_type, hash_t> arg_hashes;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaLaunch entry\n");
    kernelTimer.kernel_start_time(cbInfo);
    // const auto params = ((cudaLaunch_v3020_params
    // *)(cbInfo->functionParams));
    // const uintptr_t func = (uintptr_t) params->func;

    // The kernel could modify each argument value.
    // Check the hash of each argument so that when the call exits, we can see
    // if it was modified.

    // arg_hashes.clear();
    // for (const auto &argKey : kernelArgIds) {
    //   const auto &argValue = values[argKey];
    //   assert(argValue->address_space().is_cuda());
    // auto digest = hash_device(argValue->pos(), argValue->size());
    // printf("digest: %llu\n", digest);
    // arg_hashes[argKey] = digest;
    // }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    kernelTimer.kernel_end_time(cbInfo);

    auto api = std::make_shared<ApiRecord>(
        cbInfo->functionName, cbInfo->symbolName,
        DriverState::this_thread().current_device());

    // The kernel could have modified any argument values.
    // Hash each value and compare to the one recorded at kernel launch
    // If there is a difference, create a new value
    for (const auto &argValId : kernelArgIds) {
      const auto &argValue = values[argValId];
      api->add_input(argValId);
      // const auto digest = hash_device(argValue->pos(), argValue->size());

      // if (arg_hashes.count(argKey)) {
      //   printf("digest: %llu ==> %llu\n", arg_hashes[argKey], digest);
      // }
      // no recorded hash, or hash does not match => new value
      // if (arg_hashes.count(argKey) == 0 || digest != arg_hashes[argKey]) {
      Values::id_type newId;
      Values::value_type newVal;
      std::tie(newId, newVal) = values.duplicate_value(argValue);
      for (const auto &depId : kernelArgIds) {
        printf("launch: %lu deps on %lu\n", newId, depId);
        newVal->add_depends_on(depId);
      }
      api->add_output(newId);
      // }
    }
    APIs::record(api);
    ConfiguredCall().valid = false;
    ConfiguredCall().args.clear();

    kernelTimer.write_to_file();

  } else {
    assert(0 && "How did we get here?");
  }

  printf("callback: cudaLaunch: done\n");
  if (ConfiguredCall().valid)
    kernelTimer.save_configured_call(cbInfo->correlationId,
                                     ConfiguredCall().args);
}

void record_memcpy(const CUpti_CallbackData *cbInfo, Allocations &allocations,
                   Values &values, const ApiRecordRef &api, const uintptr_t dst,
                   const uintptr_t src, const MemoryCopyKind &kind,
                   const size_t count, const int peerSrc, const int peerDst) {

  AddressSpace srcAS, dstAS;
  Allocation srcAlloc(nullptr), dstAlloc(nullptr);

  // Set address space, and create missing allocations along the way
  if (MemoryCopyKind::CudaHostToDevice() == kind) {
    printf("%lu --[h2d]--> %lu\n", src, dst);

    // Look for, or create a source allocation
    // FIXME: address space depends on UVA
    srcAlloc = allocations.find(src, count, AddressSpace::Cuda());
    if (!srcAlloc) {
      Memory M(Memory::Host, get_numa_node(src));
      srcAlloc =
          allocations.new_allocation(src, count, AddressSpace::Host(), M,
                                     AllocationRecord::PageType::Unknown);
      printf("WARN: Couldn't find src alloc. Created implict host "
             "allocation=%lu.\n",
             src);
    }

    srcAS = srcAlloc->address_space();
    dstAS = AddressSpace::Cuda();
  } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
    printf("%lu --[d2h]--> %lu\n", src, dst);

    // Look for, or create a destination allocation
    dstAlloc = allocations.find(dst, count, AddressSpace::Cuda());
    if (!dstAlloc) {
      Memory M(Memory::Host, get_numa_node(dst));
      dstAlloc =
          allocations.new_allocation(src, count, AddressSpace::Host(), M,
                                     AllocationRecord::PageType::Unknown);
      printf("WARN: Couldn't find dst alloc. Created implict host "
             "allocation=%lu.\n",
             dst);
    }
    srcAS = AddressSpace::Cuda();
    dstAS = dstAlloc->address_space();
  } else if (MemoryCopyKind::CudaDeviceToDevice() == kind) {
    srcAS = AddressSpace::Cuda();
    dstAS = AddressSpace::Cuda();
  } else if (MemoryCopyKind::CudaDefault() == kind) {
    srcAS = AddressSpace::Cuda();
    dstAS = AddressSpace::Cuda();
  } else if (MemoryCopyKind::CudaPeer() == kind) {
    srcAS = AddressSpace::Cuda();
    dstAS = AddressSpace::Cuda();
  } else if (MemoryCopyKind::CudaHostToHost() == kind) {
    assert(0 && "Unimplemented");
  } else {
    assert(0 && "Unsupported MemoryCopyKind");
  }

  // Look for existing src / dst allocations.
  // Either we just made it, or it should already exist.
  if (!srcAlloc) {
    srcAlloc = allocations.find(src, count, srcAS);
    assert(srcAlloc);
  }
  if (!dstAlloc) {
    dstAlloc = allocations.find(dst, count, dstAS);
    assert(dstAlloc);
  }

  assert(srcAlloc && "Couldn't find or create src allocation");
  assert(dstAlloc && "Couldn't find or create dst allocation");
  // There may not be a source value, because it may have been initialized
  // on the host
  Values::id_type srcValId;
  bool found;
  std::tie(found, srcValId) =
      values.get_last_overlapping_value(src, count, srcAS);
  if (found) {
    printf("memcpy: found src value srcId=%lu\n", srcValId);
    if (!values[srcValId]->is_known_size()) {
      printf("WARN: source is unknown size. Setting by memcpy count\n");
      values[srcValId]->set_size(count);
    }
  } else {
    printf("WARN: creating implicit src value during memcpy\n");
    auto srcVal = std::shared_ptr<Value>(new Value(src, count, srcAlloc));
    values.insert(srcVal);
    srcValId = srcVal->Id();
  }

  // always create a new dst value
  Values::id_type dstValId;
  Values::value_type dstVal;
  std::tie(dstValId, dstVal) = values.new_value(dst, count, dstAlloc);
  dstVal->add_depends_on(srcValId);
  dstVal->record_meta_append(cbInfo->functionName);

  api->add_input(srcValId);
  api->add_output(dstValId);
  APIs::record(api);
}

static void handleCudaMemcpy(Allocations &allocations, Values &values,
                             const CUpti_CallbackData *cbInfo) {

  auto kernelTimer = KernelCallTime::instance();
  // extract API call parameters
  auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const cudaMemcpyKind kind = params->kind;
  const size_t count = params->count;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaMemcpy entry\n");

    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start));
    auto api = DriverState::this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);

    kernelTimer.kernel_start_time(cbInfo);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t endTimeStamp;
    cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
    printf("The end timestamp is %ul\n", endTimeStamp);
    // std::cout << "The end time is " << cbInfo->end_time;
    kernelTimer.kernel_end_time(cbInfo);
    printf("callback: cudaMemcpy end func exec\n");
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end));
    auto api = DriverState::this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind(kind), count, 0 /*unused*/, 0 /*unused */);

  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyAsync(Allocations &allocations, Values &values,
                                  const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpyAsync_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t count = params->count;
  const cudaMemcpyKind kind = params->kind;
  const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaMemcpyAsync entry\n");

    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start));
    auto api = DriverState::this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end));
    auto api = DriverState::this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind(kind), count, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyPeerAsync(Allocations &allocations, Values &values,
                                      const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpyPeerAsync_v4000_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const int dstDevice = params->dstDevice;
  const uintptr_t src = (uintptr_t)params->src;
  const int srcDevice = params->srcDevice;
  const size_t count = params->count;
  const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaMemcpyPeerAsync entry\n");
    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start));
    auto api = DriverState::this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end));
    auto api = DriverState::this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind::CudaPeer(), count, srcDevice, dstDevice);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocManaged(Allocations &allocations, Values &values,
                                    const CUpti_CallbackData *cbInfo) {
  auto params = ((cudaMallocManaged_v6000_params *)(cbInfo->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  const unsigned int flags = params->flags;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    printf("[cudaMallocManaged] %lu[%lu]\n", devPtr, size);

    // Create the new allocation
    Memory AM(Memory::CudaDevice, DriverState::this_thread().current_device());
    auto a = allocations.new_allocation(devPtr, size, AddressSpace::Cuda(), AM,
                                        AllocationRecord::PageType::Pageable);

    // Create the new value
    values.new_value(devPtr, size, a, false /*initialized*/);
  } else {
    assert(0 && "How did we get here?");
  }
}

void record_mallochost(Allocations &allocations, Values &values,
                       const uintptr_t ptr, const size_t size) {

  Allocation alloc;
  // Check if the allocation exists
  alloc = allocations.find(ptr, size, AddressSpace::Cuda());

  // If not, create a new one
  if (!alloc) {
    Memory AM(Memory::Host, get_numa_node(ptr)); // FIXME - is this right
    alloc = allocations.new_allocation(ptr, size, AddressSpace::Cuda(), AM,
                                       AllocationRecord::PageType::Pinned);
  }

  // Create the new value
  values.new_value(ptr, size, alloc, false /*initialized*/);
}

static void handleCudaMallocHost(Allocations &allocations, Values &values,
                                 const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMallocHost_v3020_params *)(cbInfo->functionParams));
    uintptr_t ptr = (uintptr_t)(*(params->ptr));
    const size_t size = params->size;
    printf("[cudaMallocHost] %lu[%lu]\n", ptr, size);

    if ((uintptr_t) nullptr == ptr) {
      printf("WARN: ignoring cudaMallocHost call that returned nullptr\n");
      return;
    }

    record_mallochost(allocations, values, ptr, size);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuMemHostAlloc(Allocations &allocations, Values &values,
                                 const CUpti_CallbackData *cbInfo) {

  auto &ts = DriverState::this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
      ts.parent_api()->cbid() ==
          CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020) {
    printf("WARN: skipping cuMemHostAlloc inside cudaMallocHost\n");
    return;
  }

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    auto params = ((cuMemHostAlloc_params *)(cbInfo->functionParams));
    uintptr_t pp = (uintptr_t)(*(params->pp));
    const size_t bytesize = params->bytesize;
    const int Flags = params->Flags;
    if (Flags & CU_MEMHOSTALLOC_PORTABLE) {
      // FIXME
      printf("WARN: cuMemHostAlloc with CU_MEMHOSTALLOC_PORTABLE\n");
    }
    if (Flags & CU_MEMHOSTALLOC_DEVICEMAP) {
      // FIXME
      printf("WARN: cuMemHostAlloc with CU_MEMHOSTALLOC_DEVICEMAP\n");
    }
    if (Flags & CU_MEMHOSTALLOC_WRITECOMBINED) {
      // FIXME
      printf("WARN: cuMemHostAlloc with CU_MEMHOSTALLOC_WRITECOMBINED\n");
    }
    printf("[cuMemHostAlloc] %lu[%lu]\n", pp, bytesize);

    record_mallochost(allocations, values, pp, bytesize);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFreeHost(Allocations &allocations, Values &values,
                               const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaFreeHost_v3020_params *)(cbInfo->functionParams));
    uintptr_t ptr = (uintptr_t)(params->ptr);
    cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
    printf("[cudaFreeHost] %lu\n", ptr);
    if (ret != cudaSuccess) {
      printf("WARN: unsuccessful cudaFreeHost: %s\n", cudaGetErrorString(ret));
    }
    assert(cudaSuccess == ret);
    assert(ptr &&
           "Must have been initialized by cudaMallocHost or cudaHostAlloc");

    auto alloc = allocations.find_exact(ptr, AddressSpace::Cuda());
    if (alloc) { // FIXME
      allocations.free(alloc->pos(), alloc->address_space());
    } else {
      assert(0 && "Freeing unallocated memory?");
    }
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMalloc(Allocations &allocations, Values &values,
                             const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
    uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
    const size_t size = params->size;
    printf("[cudaMalloc] %lu[%lu]\n", devPtr, size);

    // FIXME: could be an existing allocation from an instrumented driver
    // API

    // Create the new allocation
    // FIXME: need to check which address space this is in
    Memory AM =
        Memory(Memory::CudaDevice, DriverState::this_thread().current_device());

    Allocation a =
        allocations.new_allocation(devPtr, size, AddressSpace::Cuda(), AM,
                                   AllocationRecord::PageType::Pageable);
    printf("[cudaMalloc] new alloc id=%lu\n", a);

    values.insert(std::shared_ptr<Value>(
        new Value(devPtr, size, a, false /*initialized*/)));
    // auto digest = hash_device(devPtr, size);
    // printf("uninitialized digest: %llu\n", digest);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFree(Allocations &allocations, Values &values,
                           const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaFree entry\n");
    auto params = ((cudaFree_v3020_params *)(cbInfo->functionParams));
    auto devPtr = (uintptr_t)params->devPtr;
    cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
    printf("[cudaFree] %lu\n", devPtr);

    assert(cudaSuccess == ret);

    if (!devPtr) { // does nothing if passed 0
      printf("WARN: cudaFree called on 0? Does nothing.\n");
      return;
    }

    // Find the live matching allocation
    printf("Looking for %lu\n", devPtr);
    auto alloc = allocations.find_exact(devPtr, AddressSpace::Cuda());
    if (alloc) { // FIXME
      allocations.free(alloc->pos(), alloc->address_space());
    } else {
      assert(0 && "Freeing unallocated memory?"); // FIXME - could be async
    }
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetDevice(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaSetDevice entry\n");
    auto params = ((cudaSetDevice_v3020_params *)(cbInfo->functionParams));
    const int device = params->device;

    DriverState::this_thread().set_device(device);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaConfigureCall(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaConfigureCall entry\n");

    assert(!ConfiguredCall().valid && "call is already configured?\n");

    auto params = ((cudaConfigureCall_v3020_params *)(cbInfo->functionParams));
    ConfiguredCall().gridDim = params->gridDim;
    ConfiguredCall().blockDim = params->blockDim;
    ConfiguredCall().sharedMem = params->sharedMem;
    ConfiguredCall().stream = params->stream;
    ConfiguredCall().valid = true;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetupArgument(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaSetupArgument entry\n");
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbInfo->functionParams));
    const uintptr_t arg =
        (uintptr_t) * static_cast<const void *const *>(
                          params->arg); // arg is a pointer to the arg.
    // const size_t size     = params->size;
    // const size_t offset   = params->offset;

    assert(ConfiguredCall().valid);
    ConfiguredCall().args.push_back(arg);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamCreate(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    printf("callback: cudaStreamCreate entry\n");
    const auto params =
        ((cudaStreamCreate_v3020_params *)(cbInfo->functionParams));
    const cudaStream_t stream = *(params->pStream);
    printf("WARN: ignoring cudaStreamCreate\n");
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamDestroy(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaStreamCreate entry\n");
    const auto params =
        ((cudaStreamDestroy_v3020_params *)(cbInfo->functionParams));
    const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamSynchronize(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaStreamSynchronize entry\n");
    const auto params =
        ((cudaStreamSynchronize_v3020_params *)(cbInfo->functionParams));
    const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const CUpti_CallbackData *cbInfo) {
  (void)userdata;

  if (!DriverState::this_thread().is_cupti_callbacks_enabled()) {
    return;
  }

  if (counter == BUFFER_SIZE) {
    cuptiActivityFlushAll(0);
    counter = 0;
  }
  counter++;

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      // printf("tid=%d about to increase api stack\n", get_thread_id());
      DriverState::this_thread().api_enter(
          DriverState::this_thread().current_device(), domain, cbid, cbInfo);
    }
  }
  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleCudaMemcpy(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      handleCudaMemcpyAsync(Allocations::instance(), Values::instance(),
                            cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      handleCudaMemcpyPeerAsync(Allocations::instance(), Values::instance(),
                                cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleCudaMalloc(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
      handleCudaMallocHost(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
      handleCudaMallocManaged(Allocations::instance(), Values::instance(),
                              cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      handleCudaFree(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
      handleCudaFreeHost(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
      handleCudaConfigureCall(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
      handleCudaSetupArgument(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      handleCudaLaunch(Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetDevice_v3020:
      handleCudaSetDevice(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
      handleCudaStreamCreate(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
      handleCudaStreamDestroy(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
      handleCudaStreamSynchronize(cbInfo);
      break;
    default:
      // printf("skipping runtime call %s...\n", cbInfo->functionName);
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(Allocations::instance(), Values::instance(), cbInfo);
      break;
    default:
      // printf("skipping driver call %s...\n", cbInfo->functionName);
      break;
    }
  }
  default:
    break;
  }

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      // printf("tid=%d about maketo reduce api stack\n", get_thread_id());
      DriverState::this_thread().api_exit(domain, cbid, cbInfo);
    }
  }
}
