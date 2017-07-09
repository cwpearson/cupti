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

#include "allocation.hpp"
#include "allocations.hpp"
#include "check_cuda_error.hpp"
#include "driver_state.hpp"
#include "hash.hpp"
#include "memorycopykind.hpp"
#include "numa.hpp"
#include "value.hpp"
#include "values.hpp"

#define CHECK_CUPTI_ERROR(err, cuptifunc)                                      \
  if (err != CUPTI_SUCCESS) {                                                  \
    const char *errstr;                                                        \
    cuptiGetResultString(err, &errstr);                                        \
    printf("%s:%d:Error %s for CUPTI API function '%s'.\n", __FILE__,          \
           __LINE__, errstr, cuptifunc);                                       \
    exit(-1);                                                                  \
  }

void lazyStopCallbacks();
void lazyActivateCallbacks();

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

  // Get the current stream
  const cudaStream_t stream = ConfiguredCall().stream;
  const char *symbolName = cbInfo->symbolName;
  printf("launching %s\n", symbolName);

  // Find all values that are used by arguments
  std::vector<Values::key_type> kernelArgKeys;
  for (size_t argIdx = 0; argIdx < ConfiguredCall().args.size();
       ++argIdx) { // for each kernel argument
                   // printf("arg %lu, val %lu\n", argIdx, valIdx);

    // FIXME: assuming with p2p access, it could be on any device?
    const auto &kv =
        values.find_live_device(ConfiguredCall().args[argIdx], 1 /*size*/);

    const auto &key = kv.first;
    if (key != uintptr_t(nullptr)) {
      kernelArgKeys.push_back(kv.first);
      printf("found val %lu for kernel arg=%lu\n", key,
             ConfiguredCall().args[argIdx]);
    }
  }

  if (kernelArgKeys.empty()) {
    printf("WARN: didn't find any values for cudaLaunch\n");
  }
  // static std::map<Value::id_type, hash_t> arg_hashes;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaLaunch entry\n");
    // const auto params = ((cudaLaunch_v3020_params
    // *)(cbInfo->functionParams));
    // const uintptr_t func = (uintptr_t) params->func;

    // The kernel could modify each argument value.
    // Check the hash of each argument so that when the call exits, we can see
    // if it was modified.

    // arg_hashes.clear();
    for (const auto &argKey : kernelArgKeys) {
      const auto &argValue = values[argKey];
      assert(argValue->location().is_device_accessible() &&
             "Host pointer arg to cuda launch?");
      // auto digest = hash_device(argValue->pos(), argValue->size());
      // printf("digest: %llu\n", digest);
      // arg_hashes[argKey] = digest;
    }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    printf("callback: cudaLaunch exit\n");

    // The kernel could have modified any argument values.
    // Hash each value and compare to the one recorded at kernel launch
    // If there is a difference, create a new value
    for (const auto &argKey : kernelArgKeys) {
      const auto &argValue = values[argKey];

      // const auto digest = hash_device(argValue->pos(), argValue->size());

      // if (arg_hashes.count(argKey)) {
      //   printf("digest: %llu ==> %llu\n", arg_hashes[argKey], digest);
      // }
      // no recorded hash, or hash does not match => new value
      // if (arg_hashes.count(argKey) == 0 || digest != arg_hashes[argKey]) {
      const auto newValue =
          std::shared_ptr<Value>(new Value(*argValue)); // duplicate the value
      values.insert(newValue);
      for (const auto &depKey : kernelArgKeys) {
        printf("launch: %lu deps on %lu\n", newValue->Id(), depKey);
        newValue->add_depends_on(depKey);
      }
      // }
    }
    ConfiguredCall().valid = false;
    ConfiguredCall().args.clear();
  } else {
    assert(0 && "How did we get here?");
  }

  printf("callback: cudaLaunch: done\n");
}

void record_memcpy(Allocations &allocations, Values &values,
                   const uintptr_t dst, const uintptr_t src,
                   const MemoryCopyKind &kind, const size_t count,
                   const int peerSrc, const int peerDst) {

  Location srcLoc, dstLoc;
  if (MemoryCopyKind::CudaHostToDevice() == kind) {
    printf("%lu --[h2d]--> %lu\n", src, dst);
    srcLoc = Location(Location::Host, get_numa_node(src));
    dstLoc = Location(Location::CudaDevice, DriverState::current_device());
  } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
    printf("%lu --[d2h]--> %lu\n", src, dst);
    srcLoc = Location(Location::CudaDevice, DriverState::current_device());
    dstLoc = Location(Location::Host, get_numa_node(dst));
  } else if (MemoryCopyKind::CudaDeviceToDevice() == kind) {
    printf("%lu --[d2d]--> %lu\n", src, dst);
    const auto &dev = DriverState::current_device();
    srcLoc = dstLoc = Location(Location::CudaDevice, dev);
  } else if (MemoryCopyKind::CudaPeer() == kind) {
    printf("%lu --[p2p]--> %lu\n", src, dst);
    srcLoc = Location(Location::CudaDevice, peerSrc);
    dstLoc = Location(Location::CudaDevice, peerDst);
  } else if (MemoryCopyKind::CudaDefault() /* using cuda UVA, inferred from
                                              pointers */
             == kind) {
    printf("%lu --[???]--> %lu\n", src, dst);
    srcLoc = dstLoc =
        Location(Location::CudaUnified, DriverState::current_device());
  } else {
    assert(0 && "Unsupported cudaMemcpy kind");
  }

  // Look for existing src / dst allocations
  bool srcFound, dstFound;
  Allocations::key_type srcAllocId, dstAllocId;
  std::tie(srcFound, srcAllocId) = allocations.find_live(src, count, srcLoc);
  std::tie(dstFound, dstAllocId) = allocations.find_live(dst, count, dstLoc);

  // Destination or source allocation may be on the host, and might not have
  // been recorded.
  if (!dstFound) {
    printf("Didn't find dst value, dst=%lu\n", dst);
    assert(!dstLoc.is_device_accessible() &&
           "Couldn't find memcpy dst allocation made by CUDA runtime/driver, "
           "so dst should not be device accessible.");
    printf("WARN: creating implicit dst allocation during memcpy\n");
    std::shared_ptr<Allocation> a(
        new Allocation(dst, count, dstLoc, Allocation::Type::Pageable));
    allocations.insert(a);
    dstAllocId = a->Id();
    dstFound = true;
  }
  if (!srcFound) {
    printf("Didn't find src value, src=%lu\n", src);
    if (!srcLoc.is_device_accessible()) {
      printf("WARN: Couldn't find memcpy src allocation made by CUDA "
             "runtime/driver, so src should not be device accesible.\n");
    }
    printf("WARN: creating implicit src allocation during memcpy\n");
    std::shared_ptr<Allocation> a(
        new Allocation(src, count, srcLoc, Allocation::Type::Unknown));
    allocations.insert(a);
    srcAllocId = a->Id();
    srcFound = true;
  }

  // always create a new dst value
  auto dstVal = std::shared_ptr<Value>(new Value(dst, count, dstAllocId));
  auto dstPair = values.insert(dstVal);
  assert(dstPair.second && "Should have been a new value");

  // There may not be a source value, because it may have been initialized on
  // the host
  Values::key_type srcId;
  bool found;
  std::tie(found, srcId) =
      values.get_last_overlapping_value(src, count, srcLoc);
  if (found) {
    printf("memcpy: found src %lu\n", srcId);
    dstVal->add_depends_on(srcId);
    if (!values[srcId]->is_known_size()) {
      printf("WARN: source is unknown size. Setting by memcpy count\n");
      values[srcId]->set_size(count);
    }
    printf("found existing srcId %lu for %lu\n", srcId, src);
  } else {
    auto srcVal = std::shared_ptr<Value>(new Value(src, count, srcAllocId));
    values.insert(srcVal);
    srcId = srcVal->Id();
    dstVal->add_depends_on(srcId);
  }
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
    record_memcpy(allocations, values, dst, src, MemoryCopyKind(kind), count,
                  0 /*unused*/, 0 /*unused */);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    printf("callback: cudaMemcpy exit\n");
    if (cudaMemcpyHostToDevice == kind) {
      // auto digest = hash_device(dst, count);
      // printf("dst digest: %llu\n", digest);
    }
    if (cudaMemcpyDeviceToHost == kind) {
      // auto digest = hash_device(src, count);
      // printf("src digest: %llu\n", digest);
    }
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
    record_memcpy(allocations, values, dst, src, MemoryCopyKind(kind), count,
                  0 /*unused*/, 0 /*unused */);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
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
    record_memcpy(allocations, values, dst, src, MemoryCopyKind::CudaPeer(),
                  count, srcDevice, dstDevice);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
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
    std::shared_ptr<Allocation> a(new Allocation(
        devPtr, size,
        Location(Location::CudaUnified, DriverState::current_device()),
        Allocation::Type::Pageable));
    allocations.insert(a);

    // Create the new value
    values.insert(std::shared_ptr<Value>(
        new Value(devPtr, size, a->Id(), false /*initialized*/)));
  } else {
    assert(0 && "How did we get here?");
  }
}

void record_mallochost(Allocations &allocations, Values &values,
                       const uintptr_t ptr, const size_t size) {
  // Create the new allocation
  std::shared_ptr<Allocation> a(
      new Allocation(ptr, size, Location(Location::Host, get_numa_node(ptr)),
                     Allocation::Type::Pinned));
  allocations.insert(a);

  // Create the new value
  values.insert(std::shared_ptr<Value>(
      new Value(ptr, size, a->Id(), false /*initialized*/)));
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
    assert(ptr &&
           "Must have been initialized by cudaMallocHost or cudaHostAlloc");

    if (ret != cudaSuccess) {
      printf("WARN: unsuccessful cudaFreeHost: %s\n", cudaGetErrorString(ret));
    }

    // Find the live matching allocation
    bool found;
    Allocations::key_type allocId;
    std::tie(found, allocId) = allocations.find_live(
        ptr, Location(Location::CudaDevice, DriverState::current_device()));
    if (found) { // FIXME
      allocations.free(allocId);
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

    // FIXME: could be an existing allocation from an instrumented driver API

    // Create the new allocation
    std::shared_ptr<Allocation> a(new Allocation(
        devPtr, size,
        Location(Location::CudaDevice, DriverState::current_device()),
        Allocation::Type::Pageable));
    allocations.insert(a);

    values.insert(std::shared_ptr<Value>(
        new Value(devPtr, size, a->Id(), false /*initialized*/)));
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
    bool found;
    Allocations::key_type allocId;
    std::tie(found, allocId) = allocations.find_live(
        devPtr, Location(Location::CudaDevice, DriverState::current_device()));
    if (found) { // FIXME
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
    DriverState::set_device(device);
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
    DriverState::create_stream(stream);
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
      // auto name = cbInfo->functionName;
      // auto name = getCallbackName(domain, cbid);
      // printf("skipping runtime call %s...\n", name);
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(Allocations::instance(), Values::instance(), cbInfo);
      break;
    default:
      // auto name = cbInfo->functionName;
      // printf("skipping driver call %s...\n", name);
      break;
    }
  }
  default:
    break;
  }
}

// static const char *memcpyKindStr(enum cudaMemcpyKind kind) {
//   switch (kind) {
//   case cudaMemcpyHostToDevice:
//     return "HostToDevice";
//   case cudaMemcpyDeviceToHost:
//     return "DeviceToHost";
//   default:
//     break;
//   }

//   return "<unknown>";
// }

int activateCallbacks() {

  CUptiResult cuptierr;

  cuptierr = cuptiSubscribe(&SUBSCRIBER, (CUpti_CallbackFunc)callback, nullptr);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");
  cuptierr = cuptiEnableDomain(1, SUBSCRIBER, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");
  cuptierr = cuptiEnableDomain(1, SUBSCRIBER, CUPTI_CB_DOMAIN_DRIVER_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");

  // cuptierr = cuptiUnsubscribe(runtimeSubscriber);

  return 0;
}

int stopCallbacks() {
  CUptiResult cuptierr;
  cuptierr = cuptiUnsubscribe(SUBSCRIBER);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");
  return 0;
}

// stop callbacks if they are running
void lazyStopCallbacks() {
  if (SUBSCRIBER_ACTIVE) {
    stopCallbacks();
    SUBSCRIBER_ACTIVE = false;
  }
}

// start callbacks if they are not running
void lazyActivateCallbacks() {
  if (!SUBSCRIBER_ACTIVE) {
    activateCallbacks();
    SUBSCRIBER_ACTIVE = true;
  }
}

// start callbacks only the first time
void onceActivateCallbacks() {
  static bool done = false;
  if (!done) {
    printf("Activating callbacks for first time!\n");
    lazyActivateCallbacks();
    done = true;
  }
}
