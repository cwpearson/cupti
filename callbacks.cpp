#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cupti.h>

#include "allocation_record.hpp"
#include "allocations.hpp"
#include "apis.hpp"
#include "backtrace.hpp"
#include "driver_state.hpp"
#include "hash.hpp"
#include "memory.hpp"
#include "memorycopykind.hpp"
#include "numa.hpp"
#include "thread.hpp"
#include "util_cuda.hpp"
#include "util_cupti.hpp"
#include "value.hpp"
#include "values.hpp"

CUpti_SubscriberHandle SUBSCRIBER;
bool SUBSCRIBER_ACTIVE = 0;

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

static void handleCudaLaunch(Values &values, const CUpti_CallbackData *cbInfo) {
  printf("callback: cudaLaunch preamble\n");

  print_backtrace();

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
    // printf("callback: cudaLaunch entry\n");
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
    printf("callback: cudaLaunch exit\n");

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
  } else {
    assert(0 && "How did we get here?");
  }

  printf("callback: cudaLaunch: done\n");
}

Allocations::id_type best_effort_allocation(const uintptr_t p,
                                            const size_t count) {
  Allocations::id_type srcAllocId;
  auto &allocations = Allocations::instance();

  // Look in Host address space
  std::tie(srcAllocId, std::ignore) =
      allocations.find_live(p, count, AddressSpace::Host());
  if (srcAllocId != Allocations::noid) {
    return srcAllocId;
  }

  std::tie(srcAllocId, std::ignore) =
      allocations.find_live(p, count, AddressSpace::Cuda());
  if (srcAllocId != Allocations::noid) {
    return srcAllocId;
  }

  return Allocations::noid;
}

void record_memcpy(const CUpti_CallbackData *cbInfo, Allocations &allocations,
                   Values &values, const ApiRecordRef &api, const uintptr_t dst,
                   const uintptr_t src, const MemoryCopyKind &kind,
                   const size_t count, const int peerSrc, const int peerDst) {

  Allocations::id_type srcAllocId = 0, dstAllocId = 0;
  AddressSpace srcAS, dstAS;

  // Set address space, and create missing allocations along the way
  if (MemoryCopyKind::CudaHostToDevice() == kind) {
    printf("%lu --[h2d]--> %lu\n", src, dst);

    // Look for, or create a source allocation
    srcAllocId = best_effort_allocation(src, count);
    if (!srcAllocId) {
      Memory M(Memory::Host, get_numa_node(dst));
      std::tie(srcAllocId, std::ignore) =
          allocations.new_allocation(src, count, AddressSpace::Host(), M,
                                     AllocationRecord::PageType::Unknown);
      printf("WARN: Couldn't find src alloc. Created implict host "
             "allocation=%lu.\n",
             src);
    }

    srcAS = allocations.at(srcAllocId)->address_space();
    dstAS = AddressSpace::Cuda();
  } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
    printf("%lu --[d2h]--> %lu\n", src, dst);

    // Look for, or create a destination allocation
    dstAllocId = best_effort_allocation(dst, count);
    if (!dstAllocId) {
      Memory M(Memory::Host, get_numa_node(dst));
      std::tie(dstAllocId, std::ignore) =
          allocations.new_allocation(dst, count, AddressSpace::Host(), M,
                                     AllocationRecord::PageType::Unknown);
      printf("WARN: Couldn't find dst alloc. Created implict host "
             "allocation=%lu.\n",
             src);
    }

    srcAS = AddressSpace::Cuda();
    dstAS = allocations.at(dstAllocId)->address_space();
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
  if (!srcAllocId) {
    std::tie(srcAllocId, std::ignore) =
        allocations.find_live(src, count, srcAS);
    assert(srcAllocId != Allocations::noid);
  }
  if (!dstAllocId) {
    std::tie(dstAllocId, std::ignore) =
        allocations.find_live(dst, count, dstAS);
    assert(dstAllocId != Allocations::noid);
  }

  // There may not be a source value, because it may have been initialized
  // on
  // the host
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
    auto srcVal = std::shared_ptr<Value>(new Value(src, count, srcAllocId));
    values.insert(srcVal);
    srcValId = srcVal->Id();
  }

  // always create a new dst value
  Values::id_type dstValId;
  Values::value_type dstVal;
  std::tie(dstValId, dstVal) = values.new_value(dst, count, dstAllocId);
  dstVal->add_depends_on(srcValId);
  dstVal->record_meta_append(cbInfo->functionName);

  api->add_input(srcValId);
  api->add_output(dstValId);
  APIs::record(api);
}

static void handleCudaMemcpy(Allocations &allocations, Values &values,
                             const CUpti_CallbackData *cbInfo) {
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
    Allocations::id_type aId;
    std::tie(aId, std::ignore) =
        allocations.new_allocation(devPtr, size, AddressSpace::Cuda(), AM,
                                   AllocationRecord::PageType::Pageable);

    // Create the new value
    values.new_value(devPtr, size, aId, false /*initialized*/);
  } else {
    assert(0 && "How did we get here?");
  }
}

void record_mallochost(Allocations &allocations, Values &values,
                       const uintptr_t ptr, const size_t size) {

  Allocations::id_type aId;
  // Check if the allocation exists
  std::tie(aId, std::ignore) =
      allocations.find_live(ptr, size, AddressSpace::Cuda());

  // If not, create a new one
  if (aId == Allocations::noid) {
    Memory AM(Memory::Host, get_numa_node(ptr)); // FIXME - is this right
    std::tie(aId, std::ignore) =
        allocations.new_allocation(ptr, size, AddressSpace::Cuda(), AM,
                                   AllocationRecord::PageType::Pinned);
  }

  // Create the new value
  values.new_value(ptr, size, aId, false /*initialized*/);
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

    // Find the live matching allocation
    Allocations::id_type allocId;
    std::tie(allocId, std::ignore) =
        allocations.find_live(ptr, AddressSpace::Cuda());
    if (allocId != Allocations::noid) { // FIXME
      allocations.free(allocId);
    } else {
      // assert(0 && "Freeing unallocated memory?");
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
    std::shared_ptr<AllocationRecord> a(
        new AllocationRecord(devPtr, size, AddressSpace::Cuda(), AM,
                             AllocationRecord::PageType::Pageable));
    Allocations::id_type aId = allocations.insert(a).first->first;
    printf("[cudaMalloc] new alloc id=%lu\n", aId);

    values.insert(std::shared_ptr<Value>(
        new Value(devPtr, size, aId, false /*initialized*/)));
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
    Allocations::id_type allocId;
    printf("Looking for %lu\n", devPtr);
    std::tie(allocId, std::ignore) =
        allocations.find_live(devPtr, AddressSpace::Cuda());
    if (allocId != Allocations::noid) { // FIXME
      allocations.free(allocId);
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
      // printf("tid=%d about to reduce api stack\n", get_thread_id());
      DriverState::this_thread().api_exit(domain, cbid, cbInfo);
    }
  }
}

int activateCallbacks() {
  CUptiResult cuptierr;

  cuptierr = cuptiSubscribe(&SUBSCRIBER, (CUpti_CallbackFunc)callback, nullptr);
  CUPTI_CHECK(cuptierr);
  cuptierr = cuptiEnableDomain(1, SUBSCRIBER, CUPTI_CB_DOMAIN_RUNTIME_API);
  CUPTI_CHECK(cuptierr);
  cuptierr = cuptiEnableDomain(1, SUBSCRIBER, CUPTI_CB_DOMAIN_DRIVER_API);
  CUPTI_CHECK(cuptierr);
  return 0;
}

// start callbacks only the first time
void onceActivateCallbacks() {
  static bool done = false;
  if (!done) {
    printf("Activating callbacks for first time!\n");
    activateCallbacks();
    done = true;
  }
}

// static int stopCallbacks() {
//   CUptiResult cuptierr;
//   cuptierr = cuptiUnsubscribe(SUBSCRIBER);
//   CUPTI_CHECK(cuptierr, "cuptiUnsubscribe");
//   return 0;
// }
