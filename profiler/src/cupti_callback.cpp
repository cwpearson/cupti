#include <cassert>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <typeinfo>
#include <cxxabi.h>
#define quote(x) #x

#include <boost/date_time/posix_time/posix_time.hpp>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include "cprof/allocation.hpp"
#include "cprof/allocations.hpp"
#include "cprof/hash.hpp"
#include "cprof/memorycopykind.hpp"
#include "cprof/numa.hpp"
#include "cprof/util_cuda.hpp"
#include "cprof/util_cupti.hpp"
#include "cprof/value.hpp"
// #include "cprof/values.hpp"
// #include "cprof/dependencies.hpp"
#include "util/backtrace.hpp"

#include "cupti_subscriber.hpp"
// #include "kernel_time.hpp"
#include "profiler.hpp"

using cprof::Allocations;
using cprof::Value;
using cprof::model::Location;
using cprof::model::Memory;

// Function that is called when a Kernel is called
// Record timing in this
static void handleCudaLaunch(void *userdata, Allocations &allocations,
                             KernelCallTime &kernelTimer,
                             const CUpti_CallbackData *cbInfo) {
  profiler::err() << "INFO: callback: cudaLaunch preamble (tid="
                  << cprof::model::get_thread_id() << ")" << std::endl;
                
  // print_backtrace();

  // Get the current stream
  // const cudaStream_t stream =
  // profiler::driver().this_thread().configured_call().stream;
  const char *symbolName;
  if (!cbInfo->symbolName) {
    profiler::err() << "WARN: empty symbol name" << std::endl;
    symbolName = "[unknown symbol name]";
  } else {
    symbolName = cbInfo->symbolName;
  }
  profiler::err() << "launching " << symbolName << std::endl;

  // Find all values that are used by arguments
  std::vector<Value> kernelArgIds; // FIXME: this name is bad
  for (size_t argIdx = 0;
       argIdx < profiler::driver().this_thread().configured_call().args_.size();
       ++argIdx) { // for each kernel argument
                   // profiler::err() <<"arg %lu, val %lu\n", argIdx, valIdx);

    const int devId = profiler::driver().this_thread().current_device();
    auto AS = profiler::hardware().address_space(devId);

    // FIXME: assuming with p2p access, it could be on any device?

    const uintptr_t pos =
        profiler::driver().this_thread().configured_call().args_[argIdx];

    // if the arg is 0 it's not going to point at an allocation
    if (pos) {
      const auto &val = allocations.find_value(pos, 1 /*size*/, AS);

      if (val) {
        kernelArgIds.push_back(val);
        profiler::err()
            << "found val " << val.id() << " for kernel arg="
            << profiler::driver().this_thread().configured_call().args_[argIdx]
            << std::endl;
      }
    }
  }

  if (kernelArgIds.empty()) {
    profiler::err() << "WARN: didn't find any values for cudaLaunch"
                    << std::endl;
  }
  // static std::map<Value::id_type, hash_t> arg_hashes;

  auto api = profiler::driver().this_thread().current_api();

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "callback: cudaLaunch entry" << std::endl;

    assert(cbInfo);
    // assert(cbInfo->functionName);
    // assert(symbolName);
    // assert(cbInfo->symbolName);
    // assert(cbInfo->context);

    // uint64_t start;
    // CUPTI_CHECK(cuptiDeviceGetTimestamp(cbInfo->context, &start), std::cerr);
    api->start_ = std::chrono::high_resolution_clock::now();

    kernelTimer.kernel_start_time(cbInfo);
    // const auto params = ((cudaLaunch_v3020_params*)(cbInfo->functionParams));
    // const uintptr_t func = (uintptr_t) params->func;

    // The kernel could modify each argument value.
    // Check the hash of each argument so that when the call exits, we can see
    // if it was modified.

    // arg_hashes.clear();
    // for (const auto &argKey : kernelArgIds) {
    //   const auto &argValue = values[argKey];
    //   assert(argValue->address_space().is_cuda());
    // auto digest = hash_device(argValue->pos(), argValue->size());
    // profiler::err() <<"digest: %llu\n", digest);
    // arg_hashes[argKey] = digest;
    // }

    // for (const auto &argKey : kernelArgIds) {
    //   // const auto &argValue = values[argKey];
    //   int status;
    //   char * demangled = abi::__cxa_demangle(typeid(argKey).name(),0,0,&status);
    //   std::cout<<"Demangled value: " << demangled<<"\t"<< quote(argKey) <<"\n";
    //   // std::cout << typeof(argKey) << std::endl;
    //   // if (std::is_pointer<>::value) {
    //     // std::cout << "No" << std::endl;
    //   // }
    // }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    kernelTimer.kernel_end_time(cbInfo);

    api->end_ = std::chrono::high_resolution_clock::now();

    // The kernel could have modified any argument values.
    // Hash each value and compare to the one recorded at kernel launch
    // If there is a difference, create a new value
    for (const auto &argValue : kernelArgIds) {
      api->add_input(argValue);
      // const auto digest = hash_device(argValue->pos(), argValue->size());

      // if (arg_hashes.count(argKey)) {
      //   profiler::err() <<"digest: %llu ==> %llu\n", arg_hashes[argKey],
      //   digest);
      // }
      // no recorded hash, or hash does not match => new value
      // if (arg_hashes.count(argKey) == 0 || digest != arg_hashes[argKey]) {
      auto newVal = allocations.duplicate_value(argValue, true /*initialized*/);
      profiler::err()
          << "WARN: assuming new values from kernel launch are initialized."
          << std::endl;
      for (const auto &depVal : kernelArgIds) {
        profiler::err() << "INFO: launch: val id=" << newVal.id() << " deps on "
                        << depVal.id() << std::endl;
        newVal.add_depends_on(depVal);
      }
      api->add_output(newVal);
    }
    profiler::atomic_out(api->json());
    profiler::driver().this_thread().configured_call().valid_ = false;
    profiler::driver().this_thread().configured_call().args_.clear();
  } else {
    assert(0 && "How did we get here?");
  }

  profiler::err() << "callback: cudaLaunch: done" << std::endl;
  if (profiler::driver().this_thread().configured_call().valid_)
    kernelTimer.save_configured_call(
        cbInfo->correlationId,
        profiler::driver().this_thread().configured_call().args_);
}

static void handleCudaLaunchKernel(void *userdata, Allocations &allocations,
                                   KernelCallTime &kernelTimer,
                                   const CUpti_CallbackData *cbInfo) {
  profiler::err() << "INFO: callback: cudaLaunchKernel preamble (tid="
                  << cprof::model::get_thread_id() << ")" << std::endl;

  auto params = ((cudaLaunchKernel_v7000_params *)(cbInfo->functionParams));
  const void *func = params->func;
  profiler::err() << "launching " << func << std::endl;
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
    kernelTimer.kernel_start_time(cbInfo);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    kernelTimer.kernel_end_time(cbInfo);

    auto api = std::make_shared<ApiRecord>(
        cbInfo->functionName, cbInfo->symbolName,
        profiler::driver().this_thread().current_device());

    profiler::atomic_out(api->json());

  } else {
    assert(0 && "How did we get here?");
  }

  profiler::err() << "callback: cudaLaunchKernel: done" << std::endl;
}

void record_memcpy(const CUpti_CallbackData *cbInfo, Allocations &allocations,
                   const ApiRecordRef &api, const uintptr_t dst,
                   const uintptr_t src, const MemoryCopyKind &kind,
                   const size_t srcCount, const size_t dstCount,
                   const int peerSrc, const int peerDst) {

  (void)peerSrc;
  (void)peerDst;

  Allocation srcAlloc, dstAlloc;

  const int devId = profiler::driver().this_thread().current_device();

  // guess the src and dst address space
  auto srcAS = AddressSpace::Invalid();
  auto dstAS = AddressSpace::Invalid();
  if (profiler::hardware().cuda_device(devId).unifiedAddressing_) {
    srcAS = dstAS = profiler::hardware().address_space(devId);
  } else if (MemoryCopyKind::CudaHostToDevice() == kind) {
    srcAS = AddressSpace::Host();
    dstAS = profiler::hardware().address_space(devId);
  } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
    dstAS = AddressSpace::Host();
    srcAS = profiler::hardware().address_space(devId);
  } else if (MemoryCopyKind::CudaDefault() == kind) {
    srcAS = dstAS = AddressSpace::CudaUVA();
  } else {
    assert(0 && "Unhandled MemoryCopyKind");
  }

  assert(srcAS.is_valid());
  assert(dstAS.is_valid());
  // Set address space, and create missing allocations along the way
  if (MemoryCopyKind::CudaHostToDevice() == kind) {
    profiler::err() << src << "--[h2d]--> " << dst << std::endl;

    // Source allocation may not have been created by a CUDA api
    srcAlloc = allocations.find(src, srcCount, srcAS);
    if (!srcAlloc) {
      srcAlloc = allocations.new_allocation(src, srcCount, srcAS,
                                            Memory::Unknown, Location::Host());
      profiler::err() << "WARN: Couldn't find src alloc. Created implict host "
                         "allocation= {"
                      << srcAS.str() << "}[ " << src << " , +" << srcCount
                      << " )" << std::endl;
    }
  } else if (MemoryCopyKind::CudaDeviceToHost() == kind) {
    profiler::err() << src << "--[d2h]--> " << dst << std::endl;

    // Destination allocation may not have been created by a CUDA api
    // FIXME: we may be copying only a slice of an existing allocation. if
    // it overlaps, it should be joined
    dstAlloc = allocations.find(dst, dstCount, dstAS);
    if (!dstAlloc) {
      dstAlloc = allocations.new_allocation(dst, dstCount, dstAS,
                                            Memory::Unknown, Location::Host());
      profiler::err() << "WARN: Couldn't find dst alloc. Created implict host "
                         "allocation= {"
                      << dstAS.str() << "}[ " << dst << " , +" << dstCount
                      << " )" << std::endl;
    }
  }

  // Look for existing src / dst allocations.
  // Either we just made it, or it should already exist.
  if (!srcAlloc) {
    srcAlloc = allocations.find(src, srcCount, srcAS);
    assert(srcAlloc);
  }
  if (!dstAlloc) {
    dstAlloc = allocations.find(dst, dstCount, dstAS);
    assert(dstAlloc);
  }

  assert(srcAlloc && "Couldn't find or create src allocation");
  assert(dstAlloc && "Couldn't find or create dst allocation");
  // There may not be a source value, because it may have been initialized
  // on the host
  auto srcVal = allocations.find_value(src, srcCount, srcAlloc.address_space());
  assert(srcVal && "Value should have been created with allocation");
  profiler::err() << "memcpy: found src value srcId=" << srcVal.id()
                  << std::endl;
  profiler::err() << "WARN: Setting srcVal size by memcpy count" << std::endl;
  srcVal.set_size(srcCount);

  // always create a new dst value
  assert(srcVal);
  auto dstVal = dstAlloc.new_value(dst, dstCount, srcVal.initialized());
  assert(dstVal);
  dstVal.add_depends_on(srcVal);
  // dstVal->record_meta_append(cbInfo->functionName); // FIXME

  api->add_input(srcVal);
  api->add_output(dstVal);
  api->add_kv("kind", kind.str());
  api->add_kv("srcCount", srcCount);
  api->add_kv("dstCount", dstCount);
  profiler::atomic_out(api->json());

  auto b = std::chrono::time_point_cast<std::chrono::nanoseconds>(api->start_)
               .time_since_epoch();

  auto span = Profiler::instance().memcpyTracer_->StartSpan(
      std::to_string(cbInfo->correlationId),
      {ChildOf(&Profiler::instance().rootSpan_->context()),
       opentracing::StartTimestamp(b)});

  // span->SetTag("Transfer size", memcpyRecord->bytes);
  // span->SetTag("Transfer type",
  // memcpy_type_to_string(memcpyRecord->copyKind)); span->SetTag("Host Thread",
  // std::to_string(threadId));

  // auto timeElapsed = memcpyRecord->end - memcpyRecord->start;
  // span->SetTag("CUPTI Duration", std::to_string(timeElapsed));
  // auto err = tracer->Inject(current_span->context(), carrier);
  auto e = std::chrono::time_point_cast<std::chrono::nanoseconds>(api->end_)
               .time_since_epoch();

  span->Finish({opentracing::FinishTimestamp(e)});
}

static void handleCudaMemcpy(Allocations &allocations,
                             KernelCallTime &kernelTimer,
                             const CUpti_CallbackData *cbInfo) {

  // extract API call parameters
  auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const cudaMemcpyKind kind = params->kind;
  const size_t count = params->count;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: callback: cudaMemcpy enter" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->start_ = std::chrono::high_resolution_clock::now();

    kernelTimer.kernel_start_time(cbInfo);

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    uint64_t endTimeStamp;
    cuptiDeviceGetTimestamp(cbInfo->context, &endTimeStamp);
    // profiler::err() << "The end timestamp is " << endTimeStamp <<
    // std::endl; std::cout << "The end time is " << cbInfo->end_time;
    kernelTimer.kernel_end_time(cbInfo);

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->end_ = std::chrono::high_resolution_clock::now();

    record_memcpy(cbInfo, allocations, api, dst, src, MemoryCopyKind(kind),
                  count, count, 0 /*unused*/, 0 /*unused */);
    profiler::err() << "INFO: callback: cudaMemcpy exit" << std::endl;


    std::map<std::string, std::string> sampleMap;
    //Sample for KernelTimer
    profiler::kernelCallTime().callback_add_annotations(cbInfo->correlationId, sampleMap);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyAsync(Allocations &allocations,
                                  const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpyAsync_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t count = params->count;
  const cudaMemcpyKind kind = params->kind;
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "callback: cudaMemcpyAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->start_ = std::chrono::high_resolution_clock::now();

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->end_ = std::chrono::high_resolution_clock::now();

    record_memcpy(cbInfo, allocations, api, dst, src, MemoryCopyKind(kind),
                  count, count, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpy2DAsync(Allocations &allocations,
                                    const CUpti_CallbackData *cbInfo) {
  // extract API call parameters
  auto params = ((cudaMemcpy2DAsync_v3020_params *)(cbInfo->functionParams));
  const uintptr_t dst = (uintptr_t)params->dst;
  const size_t dpitch = params->dpitch;
  const uintptr_t src = (uintptr_t)params->src;
  const size_t spitch = params->spitch;
  // const size_t width = params->width;
  const size_t height = params->height;
  const cudaMemcpyKind kind = params->kind;
  // const cudaStream_t stream = params->stream;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "callback: cudaMemcpy2DAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->start_ = std::chrono::high_resolution_clock::now();

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->end_ = std::chrono::high_resolution_clock::now();

    const size_t srcCount = height * spitch;
    const size_t dstCount = height * dpitch;
    record_memcpy(cbInfo, allocations, api, dst, src, MemoryCopyKind(kind),
                  srcCount, dstCount, 0 /*unused*/, 0 /*unused */);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMemcpyPeerAsync(Allocations &allocations,
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
    profiler::err() << "callback: cudaMemcpyPeerAsync entry" << std::endl;

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->start_ = std::chrono::high_resolution_clock::now();
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    auto api = profiler::driver().this_thread().current_api();
    assert(api->cb_info() == cbInfo);
    assert(api->domain() == CUPTI_CB_DOMAIN_RUNTIME_API);
    api->end_ = std::chrono::high_resolution_clock::now();

    record_memcpy(cbInfo, allocations, api, dst, src,
                  MemoryCopyKind::CudaPeer(), count, count, srcDevice,
                  dstDevice);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMallocManaged(Allocations &allocations,
                                    const CUpti_CallbackData *cbInfo) {
  auto params = ((cudaMallocManaged_v6000_params *)(cbInfo->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  // const unsigned int flags = params->flags;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    profiler::err() << "INFO: [cudaMallocManaged] " << devPtr << "[" << size
                    << "]" << std::endl;

    // Create the new allocation

    const int devId = profiler::driver().this_thread().current_device();
    const int major = profiler::hardware().cuda_device(devId).major_;
    assert(major >= 3 && "cudaMallocManaged unsupported on major < 3");
    cprof::model::Memory M;
    if (major >= 6) {
      M = cprof::model::Memory::Unified6;
    } else {
      M = cprof::model::Memory::Unified3;
    }

    auto a = allocations.new_allocation(devPtr, size, AddressSpace::CudaUVA(),
                                        M, Location::Unknown());
  } else {
    assert(0 && "How did we get here?");
  }
}

void record_mallochost(Allocations &allocations, const uintptr_t ptr,
                       const size_t size) {

  auto AM = cprof::model::Memory::Pagelocked;

  const int devId = profiler::driver().this_thread().current_device();
  auto AS = profiler::hardware().address_space(devId);

  Allocation alloc =
      allocations.new_allocation(ptr, size, AS, AM, Location::Host());
  profiler::err() << "INFO: made new mallochost @ " << ptr << std::endl;

  assert(alloc);
}

static void handleCudaMallocHost(Allocations &allocations,
                                 const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMallocHost_v3020_params *)(cbInfo->functionParams));
    uintptr_t ptr = (uintptr_t)(*(params->ptr));
    const size_t size = params->size;
    profiler::err() << "INFO: [cudaMallocHost] " << ptr << "[" << size << "]"
                    << std::endl;

    if ((uintptr_t) nullptr == ptr) {
      profiler::err()
          << "WARN: ignoring cudaMallocHost call that returned nullptr"
          << std::endl;
      return;
    }

    record_mallochost(allocations, ptr, size);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuMemHostAlloc(Allocations &allocations,
                                 const CUpti_CallbackData *cbInfo) {

  auto &ts = profiler::driver().this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
      ts.parent_api()->cbid() ==
          CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020) {
    profiler::err() << "WARN: skipping cuMemHostAlloc inside cudaMallocHost"
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
      profiler::err()
          << "WARN: cuMemHostAlloc with unhandled CU_MEMHOSTALLOC_PORTABLE"
          << std::endl;
    }
    if (Flags & CU_MEMHOSTALLOC_DEVICEMAP) {
      // FIXME
      profiler::err()
          << "WARN: cuMemHostAlloc with unhandled CU_MEMHOSTALLOC_DEVICEMAP"
          << std::endl;
    }
    if (Flags & CU_MEMHOSTALLOC_WRITECOMBINED) {
      // FIXME
      profiler::err() << "WARN: cuMemHostAlloc with unhandled "
                         "CU_MEMHOSTALLOC_WRITECOMBINED"
                      << std::endl;
    }
    profiler::err() << "INFO: [cuMemHostAlloc] " << pp << "[" << bytesize << "]"
                    << std::endl;

    record_mallochost(allocations, pp, bytesize);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuLaunchKernel(Allocations &allocations,
                                 const CUpti_CallbackData *cbInfo) {

  (void)allocations;

  auto &ts = profiler::driver().this_thread();
  if (ts.in_child_api() && ts.parent_api()->is_runtime() &&
      (ts.parent_api()->cbid() == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
       ts.parent_api()->cbid() ==
           CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
    profiler::err() << "WARN: skipping cuLaunchKernel inside cudaLaunch or "
                       "cudaLaunchKernel"
                    << std::endl;
    return;
  }

  assert(0 && "unhandled cuLaunchKernel outside of cudaLaunch!");
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: enter cuLaunchKernel" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::err() << "INFO: exit cuLaunchKernel" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuModuleGetFunction(const CUpti_CallbackData *cbInfo) {

  auto params = ((cuModuleGetFunction_params *)(cbInfo->functionParams));
  const CUfunction hfunc = *(params->hfunc);
  // const CUmodule hmod = params->hmod;
  const char *name = params->name;

  profiler::err() << "INFO: cuModuleGetFunction for " << name << " @ " << hfunc
                  << std::endl;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: enter cuModuleGetFunction" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::err() << "INFO: exit cuModuleGetFunction" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuModuleGetGlobal_v2(const CUpti_CallbackData *cbInfo) {

  // auto params = ((cuModuleGetGlobal_v2_params *)(cbInfo->functionParams));

  // const CUdeviceptr dptr = *(params->dptr);
  // assert(params->bytes);
  // const size_t bytes = *(params->bytes);
  // const CUmodule hmod = params->hmod;
  // const char *name = params->name;

  // profiler::err() << "INFO: cuModuleGetGlobal_v2 for " << name << " @ " <<
  // dptr
  // << std::endl;
  profiler::err() << "WARN: ignoring cuModuleGetGlobal_v2" << std::endl;
  return;

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: enter cuModuleGetGlobal_v2" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::err() << "INFO: exit cuModuleGetGlobal_v2" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCuCtxSetCurrent(const CUpti_CallbackData *cbInfo) {

  auto params = ((cuCtxSetCurrent_params *)(cbInfo->functionParams));
  const CUcontext ctx = params->ctx;
  const int pid = cprof::model::get_thread_id();

  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::err() << "INFO: (tid=" << pid << ") setting ctx " << ctx
                    << std::endl;
    profiler::driver().this_thread().set_context(ctx);
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaFreeHost(Allocations &allocations,
                               const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaFreeHost_v3020_params *)(cbInfo->functionParams));
    uintptr_t ptr = (uintptr_t)(params->ptr);
    cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
    profiler::err() << "INFO: [cudaFreeHost] " << ptr << std::endl;
    if (ret != cudaSuccess) {
      profiler::err() << "WARN: unsuccessful cudaFreeHost: "
                      << cudaGetErrorString(ret) << std::endl;
    }
    assert(cudaSuccess == ret);
    assert(ptr &&
           "Must have been initialized by cudaMallocHost or cudaHostAlloc");

    const int devId = profiler::driver().this_thread().current_device();
    auto AS = profiler::hardware().address_space(devId);

    auto numFreed = allocations.free(ptr, AS);
    assert(numFreed && "Freeing unallocated memory?");
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaMalloc(Allocations &allocations,
                             const CUpti_CallbackData *cbInfo) {
  const auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
  const uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
  const size_t size = params->size;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: cudaMalloc: [" << devPtr << ", +" << size
                    << ") entry" << std::endl;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {

    const cudaError_t res =
        *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
    profiler::err() << "INFO: " << res << " = cudaMalloc: [" << devPtr << ", +"
                    << size << ")" << std::endl;
    if (res != cudaSuccess) {
      profiler::err() << "WARN: cudaMalloc had an error" << std::endl;
      return;
    }

   
    // Create the new allocation
    // FIXME: need to check which address space this is in
    const int devId = profiler::driver().this_thread().current_device();
    auto AS = profiler::hardware().address_space(devId);
    auto AM = cprof::model::Memory::Pageable;

    Allocation a = allocations.new_allocation(devPtr, size, AS, AM,
                                              Location::CudaDevice(devId));
    profiler::err() << "INFO: (tid=" << cprof::model::get_thread_id()
                    << ") [cudaMalloc] new alloc=" << (uintptr_t)a.id()
<< " pos=" << a.pos() << std::endl;
 
    //Create new database allocation record
    // auto dependency_tracking = DependencyTracking::instance();
    // dependency_tracking.memory_ptr_create(a->pos());

    // auto digest = hash_device(devPtr, size);
    // profiler::err() <<"uninitialized digest: %llu\n", digest);
  } else {
    assert(0 && "how did we get here?");
  }
}

static void handleCudaFree(Allocations &allocations,
                           const CUpti_CallbackData *cbInfo) {
   if (cbInfo->callbackSite == CUPTI_API_ENTER) {
   } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
     auto params = ((cudaFree_v3020_params *)(cbInfo->functionParams));
     auto devPtr = (uintptr_t)params->devPtr;
     cudaError_t ret = *static_cast<cudaError_t *>(cbInfo->functionReturnValue);
     profiler::err() << "INFO: (tid=" << cprof::model::get_thread_id()
                     << ") [cudaFree] " << devPtr << std::endl;

     assert(cudaSuccess == ret);

     if (!devPtr) { // does nothing if passed 0
       profiler::err() << "WARN: cudaFree called on 0? Does nothing."
                       << std::endl;
       return;
     }

     const int devId = profiler::driver().this_thread().current_device();
     auto AS = profiler::hardware().address_space(devId);

     // Find the live matching allocation
     profiler::err() << "Looking for " << devPtr << std::endl;
     auto freeAlloc = allocations.free(devPtr, AS);
     assert(freeAlloc && "Freeing unallocated memory?");
   } else {
     assert(0 && "How did we get here?");
   }
}

static void handleCudaSetDevice(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "callback: cudaSetDevice entry" << std::endl;
    auto params = ((cudaSetDevice_v3020_params *)(cbInfo->functionParams));
    const int device = params->device;

    profiler::driver().this_thread().set_device(device);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaConfigureCall(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: ( tid=" << cprof::model::get_thread_id()
                    << " ) callback: cudaConfigureCall entry" << std::endl;

    assert(!profiler::driver().this_thread().configured_call().valid_ &&
           "call is already configured?\n");

    auto params = ((cudaConfigureCall_v3020_params *)(cbInfo->functionParams));
    profiler::driver().this_thread().configured_call().gridDim_ =
        params->gridDim;
    profiler::driver().this_thread().configured_call().blockDim_ =
        params->blockDim;
    profiler::driver().this_thread().configured_call().sharedMem_ =
        params->sharedMem;
    profiler::driver().this_thread().configured_call().stream_ = params->stream;
    profiler::driver().this_thread().configured_call().valid_ = true;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaSetupArgument(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "callback: cudaSetupArgument entry (tid="
                    << cprof::model::get_thread_id() << ")" << std::endl;
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbInfo->functionParams));
    const uintptr_t arg =
        (uintptr_t) * static_cast<const void *const *>(
                          params->arg); // arg is a pointer to the arg.
    // const size_t size     = params->size;
    // const size_t offset   = params->offset;

    assert(profiler::driver().this_thread().configured_call().valid_);
    profiler::driver().this_thread().configured_call().args_.push_back(arg);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamCreate(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    profiler::err() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    // const auto params =
    //     ((cudaStreamCreate_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = *(params->pStream);
    profiler::err() << "WARN: ignoring cudaStreamCreate" << std::endl;
  } else {
    assert(0 && "How did we get here?");
  }
}

static void handleCudaStreamDestroy(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    profiler::err() << "INFO: callback: cudaStreamCreate entry" << std::endl;
    profiler::err() << "WARN: ignoring cudaStreamDestroy" << std::endl;
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
    profiler::err() << "INFO: callback: cudaStreamSynchronize entry"
                    << std::endl;
    profiler::err() << "WARN: ignoring cudaStreamSynchronize" << std::endl;
    // const auto params =
    //     ((cudaStreamSynchronize_v3020_params *)(cbInfo->functionParams));
    // const cudaStream_t stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

void CUPTIAPI cuptiCallbackFunction(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    const CUpti_CallbackData *cbInfo) {
  (void)userdata; // data supplied at subscription

  if (!profiler::driver().this_thread().is_cupti_callbacks_enabled()) {
    return;
  }

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) ||
      (domain == CUPTI_CB_DOMAIN_RUNTIME_API)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      profiler::driver().this_thread().api_enter(
          profiler::driver().this_thread().current_device(), domain, cbid,
          cbInfo);
    }
  }
  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleCudaMemcpy(profiler::allocations(), profiler::kernelCallTime(),
                       cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
      handleCudaMemcpyAsync(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyPeerAsync_v4000:
      handleCudaMemcpyPeerAsync(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleCudaMalloc(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020:
      handleCudaMallocHost(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000:
      handleCudaMallocManaged(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
      handleCudaFree(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020:
      handleCudaFreeHost(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaConfigureCall_v3020:
      handleCudaConfigureCall(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaSetupArgument_v3020:
      handleCudaSetupArgument(cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      handleCudaLaunch(userdata, profiler::allocations(), profiler::kernelCallTime(),
                       cbInfo);
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
      handleCudaMemcpy2DAsync(profiler::allocations(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      handleCudaLaunchKernel(userdata, profiler::allocations(),
                             profiler::kernelCallTime(), cbInfo);
      break;
    default:
      profiler::err() << "DEBU: ( tid= " << cprof::model::get_thread_id()
                      << " ) skipping runtime call " << cbInfo->functionName
                      << std::endl;
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      handleCuMemHostAlloc(profiler::allocations(), cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
      handleCuLaunchKernel(profiler::allocations(), cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
      handleCuModuleGetFunction(cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuModuleGetGlobal_v2:
      handleCuModuleGetGlobal_v2(cbInfo);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSetCurrent:
      handleCuCtxSetCurrent(cbInfo);
      break;
    default:
      profiler::err() << "DEBU: ( tid= " << cprof::model::get_thread_id()
                      << " ) skipping driver call " << cbInfo->functionName
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
      profiler::driver().this_thread().api_exit(domain, cbid, cbInfo);
    }
  }
}
