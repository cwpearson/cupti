#include <cassert>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include "cprof/activity_callbacks.hpp"
#include "cprof/allocation.hpp"
#include "cprof/allocations.hpp"
#include "cprof/apis.hpp"
#include "cprof/hash.hpp"
#include "cprof/kernel_time.hpp"
#include "cprof/memorycopykind.hpp"
#include "cprof/numa.hpp"
#include "cprof/profiler.hpp"
#include "cprof/util_cuda.hpp"
#include "cprof/util_cupti.hpp"
#include "cprof/value.hpp"
#include "cprof/values.hpp"
#include "util/backtrace.hpp"

using cprof::model::Location;
using cprof::model::Memory;

// Function that is called when a Kernel is called
// Record timing in this
static void handleCudaLaunch(Values &values, const CUpti_CallbackData *cbInfo) {
  cprof::err() << "INFO: callback: cudaLaunch preamble (tid="
               << cprof::model::get_thread_id() << ")" << std::endl;
  auto kernelTimer = KernelCallTime::instance();

  // print_backtrace();

  // Get the current stream
  // const cudaStream_t stream =
  // cprof::driver().this_thread().configured_call().stream;
  const char *symbolName = cbInfo->symbolName;
  cprof::err() << "launching " << symbolName << std::endl;

  // Find all values that are used by arguments
  std::vector<Value> kernelArgIds; // FIXME: this name is bad
  for (size_t argIdx = 0;
       argIdx < cprof::driver().this_thread().configured_call().args_.size();
       ++argIdx) { // for each kernel argument
                   // cprof::err() <<"arg %lu, val %lu\n", argIdx, valIdx);

    const int devId = cprof::driver().this_thread().current_device();
    auto AS = cprof::hardware().address_space(devId);

    // FIXME: assuming with p2p access, it could be on any device?
    const auto &val = values.find_live(
        cprof::driver().this_thread().configured_call().args_[argIdx],
        1 /*size*/, AS);

    if (val) {
      kernelArgIds.push_back(val);
      cprof::err()
          << "found val " << val << " for kernel arg="
          << cprof::driver().this_thread().configured_call().args_[argIdx]
          << std::endl;
    }
  }

  if (kernelArgIds.empty()) {
    cprof::err() << "WARN: didn't find any values for cudaLaunch" << std::endl;
  }
  // static std::map<Value::id_type, hash_t> arg_hashes;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaLaunch entry" << std::endl;
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
    // cprof::err() <<"digest: %llu\n", digest);
    // arg_hashes[argKey] = digest;
    // }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    kernelTimer.kernel_end_time(cbInfo);

    auto api = std::make_shared<ApiRecord>(
        cbInfo->functionName, cbInfo->symbolName,
        cprof::driver().this_thread().current_device());

    // The kernel could have modified any argument values.
    // Hash each value and compare to the one recorded at kernel launch
    // If there is a difference, create a new value
    for (const auto &argValue : kernelArgIds) {
      api->add_input(argValue);
      // const auto digest = hash_device(argValue->pos(), argValue->size());

      // if (arg_hashes.count(argKey)) {
      //   cprof::err() <<"digest: %llu ==> %llu\n", arg_hashes[argKey],
      //   digest);
      // }
      // no recorded hash, or hash does not match => new value
      // if (arg_hashes.count(argKey) == 0 || digest != arg_hashes[argKey]) {
      auto newVal = values.duplicate_value(argValue);
      for (const auto &depVal : kernelArgIds) {
        cprof::err() << "INFO: launch: " << newVal << " deps on " << depVal
                     << std::endl;
        newVal.add_depends_on(depVal);
      }
      api->add_output(newVal);
    }
    APIs::record(api);
    cprof::driver().this_thread().configured_call().valid_ = false;
    cprof::driver().this_thread().configured_call().args_.clear();
  } else {
    assert(0 && "How did we get here?");
  }

  cprof::err() << "callback: cudaLaunch: done" << std::endl;
  if (cprof::driver().this_thread().configured_call().valid_)
    kernelTimer.save_configured_call(
        cbInfo->correlationId,
        cprof::driver().this_thread().configured_call().args_);
}

static void handleCudaLaunchKernel(Values &values,
                                   const CUpti_CallbackData *cbInfo) {
  cprof::err() << "INFO: callback: cudaLaunchKernel preamble (tid="
               << cprof::model::get_thread_id() << ")" << std::endl;
  auto kernelTimer = KernelCallTime::instance();

  auto params = ((cudaLaunchKernel_v7000_params *)(cbInfo->functionParams));
  const void *func = params->func;
  cprof::err() << "launching " << func << std::endl;
  const dim3 gridDim = params->gridDim;
  const dim3 blockDim = params->blockDim;
  void *const *args = params->args;
  const size_t sharedMem = params->sharedMem;
  const cudaStream_t stream = params->stream;

  // print_backtrace();

  const char *symbolName = cbInfo->symbolName;
  // const char *symbolName = (char*)func;

  assert(0 && "Unimplemented");

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaLaunch entry" << std::endl;
    kernelTimer.kernel_start_time(cbInfo);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    kernelTimer.kernel_end_time(cbInfo);

    auto api = std::make_shared<ApiRecord>(
        cbInfo->functionName, cbInfo->symbolName,
        cprof::driver().this_thread().current_device());

    APIs::record(api);

  } else {
    assert(0 && "How did we get here?");
  }

  cprof::err() << "callback: cudaLaunchKernel: done" << std::endl;
}

void record_memcpy(const CUpti_CallbackData *cbInfo, Allocations &allocations,
                   Values &values, const ApiRecordRef &api, const uintptr_t dst,
                   const uintptr_t src, const MemoryCopyKind &kind,
                   const size_t srcCount, const size_t dstCount,
                   const int peerSrc, const int peerDst) {

  Allocation srcAlloc(nullptr), dstAlloc(nullptr);

  const int devId = cprof::driver().this_thread().current_device();

  // Set address space, and create missing allocations along the way
  if (MemoryCopyKind::CudaHostToDevice() == kind) {
    cprof::err() << src << "--[h2d]--> " << dst << std::endl;

    // Source allocation may not have been created by a CUDA api
    srcAlloc = allocations.find(src, srcCount);
    if (!srcAlloc) {
      const auto AS = cprof::hardware().address_space(devId);
      srcAlloc = allocations.new_allocation(src, srcCount, AS, Memory::Unknown,
                                            Location::Host());
      cprof::err() << "WARN: Couldn't find src alloc. Created implict host "
                      "allocation= [ "
                   << src << " , + " << srcCount << " )" << std::endl;
    }
  } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
    cprof::err() << src << "--[d2h]--> " << dst << std::endl;

    // Destination allocation may not have been created by a CUDA api
    // FIXME: we may be copying only a slice of an existing allocation. if
    // it overlaps, it should be joined
    dstAlloc = allocations.find(dst, dstCount);
    if (!dstAlloc) {
      const auto AS = cprof::hardware().address_space(devId);
      dstAlloc = allocations.new_allocation(dst, dstCount, AS, Memory::Unknown,
                                            Location::Host());
      cprof::err() << "WARN: Couldn't find dst alloc. Created implict host "
                      "allocation= [ "
                   << dst << " , + " << dstCount << " )" << std::endl;
    }
  } else if (MemoryCopyKind::CudaDefault() == kind) {
    srcAlloc = srcAlloc =
        allocations.find(src, srcCount, AddressSpace::CudaUVA());
    srcAlloc = srcAlloc =
        allocations.find(dst, dstCount, AddressSpace::CudaUVA());
  }

  // Look for existing src / dst allocations.
  // Either we just made it, or it should already exist.
  if (!srcAlloc) {
    srcAlloc = allocations.find(src, srcCount);
    assert(srcAlloc);
  }
  if (!dstAlloc) {
    dstAlloc = allocations.find(dst, dstCount);
    assert(dstAlloc);
  }

  assert(srcAlloc && "Couldn't find or create src allocation");
  assert(dstAlloc && "Couldn't find or create dst allocation");
  // There may not be a source value, because it may have been initialized
  // on the host
  auto srcVal = values.find_live(src, srcCount, srcAlloc->address_space());
  if (srcVal) {
    cprof::err() << "memcpy: found src value srcId=" << srcVal << std::endl;
    cprof::err() << "WARN: Setting srcVal size by memcpy count" << std::endl;
    srcVal->set_size(srcCount);
  } else {
    cprof::err() << "WARN: creating implicit src value during memcpy"
                 << std::endl;
    srcVal = values.new_value(src, srcCount, srcAlloc, true /*initialized*/);
  }

  // always create a new dst value
  assert(srcVal);
  auto dstVal =
      values.new_value(dst, dstCount, dstAlloc, srcVal->initialized());
  assert(dstVal);
  dstVal.add_depends_on(srcVal);
  // dstVal->record_meta_append(cbInfo->functionName); // FIXME

  api->add_input(srcVal);
  api->add_output(dstVal);
  api->add_kv("kind", kind.str());
  api->add_kv("srcCount", srcCount);
  api->add_kv("dstCount", dstCount);
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
    cprof::err() << "INFO: callback: cudaMemcpy enter" << std::endl;
    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);

    kernelTimer.kernel_start_time(cbInfo);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t endTimeStamp;
    cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
    // cprof::err() << "The end timestamp is " << endTimeStamp << std::endl;
    // std::cout << "The end time is " << cbInfo->end_time;
    kernelTimer.kernel_end_time(cbInfo);
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind(kind), count, count, 0 /*unused*/,
                  0 /*unused */);
    cprof::err() << "INFO: callback: cudaMemcpy exit" << std::endl;

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
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaMemcpyAsync entry" << std::endl;

    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind(kind), count, count, 0 /*unused*/,
                  0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpy2DAsync(Allocations &allocations, Values &values,
                                    const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpy2DAsync_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const size_t dpitch = params->dpitch;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t spitch = params->spitch;
  const size_t width = params->width;
  const size_t height = params->height;
  const cudaMemcpyKind kind = params->kind;
  const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaMemcpy2DAsync entry" << std::endl;

    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    const size_t srcCount = height * spitch;
    const size_t dstCount = height * dpitch;
    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind(kind), srcCount, dstCount, 0 /*unused*/,
                  0 /*unused */);
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
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaMemcpyPeerAsync entry" << std::endl;
    uint64_t start;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_start_time(start);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t end;
    CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &end), cprof::err());
    auto api = cprof::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->record_end_time(end);

    record_memcpy(cbInfo, allocations, values, api, dst, src,
                  MemoryCopyKind::CudaPeer(), count, count, srcDevice,
                  dstDevice);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocManaged(Allocations &allocations, Values &values,
                                    const CUpti_CallbackData *cbInfo) {
  auto params = ((cudaMallocManaged_v6000_params *)(cbInfo->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  // const unsigned int flags = params->flags;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    cprof::err() << "INFO: [cudaMallocManaged] " << devPtr << "[" << size << "]"
                 << std::endl;

    // Create the new allocation

    const int devId = cprof::driver().this_thread().current_device();
    const int major = cprof::hardware().cuda_device(devId).major_;
    assert(major >= 3 && "cudaMallocManaged unsupported on major < 3");
    cprof::model::Memory M;
    if (major >= 6) {
      M = cprof::model::Memory::Unified6;
    } else {
      M = cprof::model::Memory::Unified3;
    }

    auto a = allocations.new_allocation(devPtr, size, AddressSpace::CudaUVA(),
                                        M, Location::Unknown());

    // Create the new value
    values.new_value(devPtr, size, a, false /*initialized*/);
  } else {
    assert(0 && "How did we get here?");
  }
}

void record_mallochost(Allocations &allocations, Values &values,
                       const uintptr_t ptr, const size_t size) {

  auto AM = cprof::model::Memory::Pagelocked;

  const int devId = cprof::driver().this_thread().current_device();
  auto AS = cprof::hardware().address_space(devId);

  Allocation alloc = allocations.find(ptr, size, AS);
  if (!alloc) {
    alloc = allocations.new_allocation(ptr, size, AS, AM, Location::Host());
  }

  values.new_value(ptr, size, alloc, false /*initialized*/);
}

static void handleCudaMallocHost(Allocations &allocations, Values &values,
                                 const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMallocHost_v3020_params *)(cbInfo->functionParams));
    uintptr_t ptr = (uintptr_t)(*(params->ptr));
    const size_t size = params->size;
    cprof::err() << "INFO: [cudaMallocHost] " << ptr << "[" << size << "]"
                 << std::endl;

    if ((uintptr_t) nullptr == ptr) {
      cprof::err() << "WARN: ignoring cudaMallocHost call that returned nullptr"
                   << std::endl;
      return;
    }

    record_mallochost(allocations, values, ptr, size);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuMemHostAlloc(Allocations &allocations, Values &values,
                                 const CUpti_CallbackData *cbInfo) {

  auto &ts = cprof::driver().this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
      ts.parent_api()->cbid() ==
          CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020) {
    cprof::err() << "WARN: skipping cuMemHostAlloc inside cudaMallocHost"
                 << std::endl;
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
      cprof::err()
          << "WARN: cuMemHostAlloc with unhandled CU_MEMHOSTALLOC_PORTABLE"
          << std::endl;
    }
    if (Flags & CU_MEMHOSTALLOC_DEVICEMAP) {
      // FIXME
      cprof::err()
          << "WARN: cuMemHostAlloc with unhandled CU_MEMHOSTALLOC_DEVICEMAP"
          << std::endl;
    }
    if (Flags & CU_MEMHOSTALLOC_WRITECOMBINED) {
      // FIXME
      cprof::err() << "WARN: cuMemHostAlloc with unhandled "
                      "CU_MEMHOSTALLOC_WRITECOMBINED"
                   << std::endl;
    }
    cprof::err() << "INFO: [cuMemHostAlloc] " << pp << "[" << bytesize << "]"
                 << std::endl;

    record_mallochost(allocations, values, pp, bytesize);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuLaunchKernel(Values &values,
                                 const CUpti_CallbackData *cbInfo) {

  auto &ts = cprof::driver().this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
          ts.parent_api()->cbid() ==
              CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
      ts.parent_api()->cbid() ==
          CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
    cprof::err()
        << "WARN: skipping cuLaunchKernel inside cudaLaunch or cudaLaunchKernel"
        << std::endl;
    return;
  }

  assert(0 && "unhandled cuLaunchKernel outside of cudaLaunch!");
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: enter cuLaunchKernel" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    cprof::err() << "INFO: exit cuLaunchKernel" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuModuleGetFunction(const CUpti_CallbackData *cbInfo) {

  auto params = ((cuModuleGetFunction_params *)(cbInfo->functionParams));
  const CUfunction hfunc = *(params->hfunc);
  const CUmodule hmod = params->hmod;
  const char *name = params->name;

  cprof::err() << "INFO: cuModuleGetFunction for " << name << " @ " << hfunc
               << std::endl;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: enter cuModuleGetFunction" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    cprof::err() << "INFO: exit cuModuleGetFunction" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuModuleGetGlobal_v2(const CUpti_CallbackData *cbInfo) {

  auto params = ((cuModuleGetGlobal_v2_params *)(cbInfo->functionParams));

  const CUdeviceptr dptr = *(params->dptr);
  // assert(params->bytes);
  // const size_t bytes = *(params->bytes);
  const CUmodule hmod = params->hmod;
  const char *name = params->name;

  // cprof::err() << "INFO: cuModuleGetGlobal_v2 for " << name << " @ " << dptr
  // << std::endl;
  cprof::err() << "WARN: ignoring cuModuleGetGlobal_v2" << std::endl;
  return;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: enter cuModuleGetGlobal_v2" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    cprof::err() << "INFO: exit cuModuleGetGlobal_v2" << std::endl;
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
    cprof::err() << "INFO: [cudaFreeHost] " << ptr << std::endl;
    if (ret != cudaSuccess) {
      cprof::err() << "WARN: unsuccessful cudaFreeHost: "
                   << cudaGetErrorString(ret) << std::endl;
    }
    assert(cudaSuccess == ret);
    assert(ptr &&
           "Must have been initialized by cudaMallocHost or cudaHostAlloc");

    const int devId = cprof::driver().this_thread().current_device();
    auto AS = cprof::hardware().address_space(devId);

    auto alloc = allocations.find_exact(ptr, AS);
    if (alloc) {
      assert(allocations.free(alloc->pos(), alloc->address_space()) &&
             "memory not freed");
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
    cprof::err() << "INFO: [cudaMalloc] " << devPtr << "[" << size << "]"
                 << std::endl;

    // FIXME: could be an existing allocation from an instrumented driver
    // API

    // Create the new allocation
    // FIXME: need to check which address space this is in
    const int devId = cprof::driver().this_thread().current_device();
    auto AS = cprof::hardware().address_space(devId);
    auto AM = cprof::model::Memory::Pageable;

    Allocation a = allocations.new_allocation(devPtr, size, AS, AM,
                                              Location::CudaDevice(devId));
    cprof::err() << "INFO: [cudaMalloc] new alloc=" << (uintptr_t)a.get()
                 << " pos=" << a->pos() << std::endl;

    values.new_value(devPtr, size, a, false /*initialized*/);
    // auto digest = hash_device(devPtr, size);
    // cprof::err() <<"uninitialized digest: %llu\n", digest);
  } else {
    assert(0 && "how did we get here?");
  }
}

static void handleCudaFree(Allocations &allocations, Values &values,
                           const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: callback: cudaFree entry" << std::endl;
    auto params = ((cudaFree_v3020_params *)(cbInfo->functionParams));
    auto devPtr = (uintptr_t)params->devPtr;
    cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
    cprof::err() << "INFO: [cudaFree] " << devPtr << std::endl;

    assert(cudaSuccess == ret);

    if (!devPtr) { // does nothing if passed 0
      cprof::err() << "WARN: cudaFree called on 0? Does nothing." << std::endl;
      return;
    }

    const int devId = cprof::driver().this_thread().current_device();
    auto AS = cprof::hardware().address_space(devId);

    // Find the live matching allocation
    cprof::err() << "Looking for " << devPtr << std::endl;
    auto alloc = allocations.find_exact(devPtr, AS);
    if (alloc) { // FIXME
      assert(allocations.free(alloc->pos(), alloc->address_space()));
    } else {
      cprof::err() << "ERR: Freeing unallocated memory?"
                   << std::endl; // FIXME - could be async
    }
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetDevice(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaSetDevice entry" << std::endl;
    auto params = ((cudaSetDevice_v3020_params *)(cbInfo->functionParams));
    const int device = params->device;

    cprof::driver().this_thread().set_device(device);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaConfigureCall(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: callback: cudaConfigureCall entry (tid="
                 << cprof::model::get_thread_id() << std::endl;

    assert(!cprof::driver().this_thread().configured_call().valid_ &&
           "call is already configured?\n");

    auto params = ((cudaConfigureCall_v3020_params *)(cbInfo->functionParams));
    cprof::driver().this_thread().configured_call().gridDim_ = params->gridDim;
    cprof::driver().this_thread().configured_call().blockDim_ =
        params->blockDim;
    cprof::driver().this_thread().configured_call().sharedMem_ =
        params->sharedMem;
    cprof::driver().this_thread().configured_call().stream_ = params->stream;
    cprof::driver().this_thread().configured_call().valid_ = true;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetupArgument(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "callback: cudaSetupArgument entry (tid="
                 << cprof::model::get_thread_id() << ")\n";
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbInfo->functionParams));
    const uintptr_t arg =
        (uintptr_t) * static_cast<const void *const *>(
                          params->arg); // arg is a pointer to the arg.
    // const size_t size     = params->size;
    // const size_t offset   = params->offset;

    assert(cprof::driver().this_thread().configured_call().valid_);
    cprof::driver().this_thread().configured_call().args_.push_back(arg);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamCreate(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    cprof::err() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    // const auto params =
    //     ((cudaStreamCreate_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = *(params->pStream);
    cprof::err() << "WARN: ignoring cudaStreamCreate" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamDestroy(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    cprof::err() << "WARN: ignoring cudaStreamDestroy" << std::endl;
    // const auto params =
    //     ((cudaStreamDestroy_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamSynchronize(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    cprof::err() << "INFO: callback: cudaStreamSynchronize entry" << std::endl;
    cprof::err() << "WARN: ignoring cudaStreamSynchronize" << std::endl;
    // const auto params =
    //     ((cudaStreamSynchronize_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid,
                       const CUpti_CallbackData *cbInfo) {
  (void)userdata;

  if (!cprof::driver().this_thread().is_cupti_callbacks_enabled()) {
    return;
  }

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      cprof::driver().this_thread().api_enter(
          cprof::driver().this_thread().current_device(), domain, cbid, cbInfo);
    }
  }
  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleCudaMemcpy(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      handleCudaMemcpyAsync(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      handleCudaMemcpyPeerAsync(cprof::allocations(), Values::instance(),
                                cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleCudaMalloc(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
      handleCudaMallocHost(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
      handleCudaMallocManaged(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      handleCudaFree(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
      handleCudaFreeHost(cprof::allocations(), Values::instance(), cbInfo);
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
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020:
      handleCudaMemcpy2DAsync(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      handleCudaLaunchKernel(Values::instance(), cbInfo);
      break;
    default:
      cprof::err() << "DEBU: skipping runtime call " << cbInfo->functionName
                   << std::endl;
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(cprof::allocations(), Values::instance(), cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      handleCuLaunchKernel(Values::instance(), cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
      handleCuModuleGetFunction(cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2:
      handleCuModuleGetGlobal_v2(cbInfo);
      break;
    default:
      cprof::err() << "DEBU: skipping driver call " << cbInfo->functionName
                   << std::endl;
      break;
    }
  }
  default:
    break;
  }

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      // cprof::err() <<"tid=%d about maketo reduce api stack\n",
      // get_thread_id());
      cprof::driver().this_thread().api_exit(domain, cbid, cbInfo);
    }
  }
}
