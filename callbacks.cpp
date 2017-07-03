#include <array>
#include <cassert>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cupti.h>

#include "allocation.hpp"
#include "allocations.hpp"
#include "check_cuda_error.hpp"
#include "numa.hpp"
#include "set_device.hpp"
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

void handleCudaSetDevice(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaSetDevice entry\n");
    auto params = ((cudaSetDevice_v3020_params *)(cbInfo->functionParams));
    SetDevice().device_ = params->device;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "unexpected callbackSite");
  }
}

void handleMemcpy(Allocations &allocations, Values &values,
                  const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaMemcpy entry\n");
    // extract API call parameters
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    const uintptr_t dst = (uintptr_t)params->dst;
    const uintptr_t src = (uintptr_t)params->src;
    const cudaMemcpyKind kind = params->kind;
    const size_t count = params->count;

    Location srcLoc, dstLoc;
    if (cudaMemcpyHostToDevice == kind) {
      printf("%lu --[h2d]--> %lu\n", src, dst);
      srcLoc = Location::Host(get_numa_node(src));
      dstLoc = Location::Device(SetDevice().current_device());
    } else if (cudaMemcpyDeviceToHost == kind) {
      printf("%lu --[d2h]--> %lu\n", src, dst);
      srcLoc = Location::Device(SetDevice().current_device());
      dstLoc = Location::Host(get_numa_node(dst));
    } else if (cudaMemcpyDeviceToDevice == kind) {
      printf("%lu --[d2d]--> %lu\n", src, dst);
      const auto &dev = SetDevice().current_device();
      srcLoc = Location::Device(dev);
      dstLoc = Location::Device(dev);
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
      assert(dstLoc.is_host() && "How did we miss this value");
      printf("WARN: creating implicit host dst allocation during memcpy\n");
      std::shared_ptr<Allocation> a(new Allocation(dst, count, dstLoc));
      allocations.insert(a);
      dstAllocId = a->Id();
      dstFound = true;
    }
    if (!srcFound) {
      assert(srcLoc.is_host() && "How did we miss this value");
      printf("WARN: creating implicit host src allocation during memcpy\n");
      std::shared_ptr<Allocation> a(new Allocation(src, count, srcLoc));
      allocations.insert(a);
      srcAllocId = a->Id();
      srcFound = true;
    }

    // always creates a new dst value
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

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    // for (auto kv : values) {
    //   const auto &valIdx = kv.first;
    //   const auto &val = kv.second;
    //   if (val->depends_on().size() > 0) {
    //     printf("%lu <- ", valIdx);
    //     for (const auto &d : val->depends_on()) {
    //       printf("%lu ", d);
    //     }
    //     printf("\n");
    //   }
    // }
  } else {
    assert(0 && "How did we get here?");
  }
}

void handleCudaMalloc(Allocations &allocations, Values &values,
                      const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
    uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
    const size_t size = params->size;
    printf("[cudaMalloc] %lu[%lu]\n", devPtr, size);

    // Create the new allocation
    std::shared_ptr<Allocation> a(new Allocation(
        devPtr, size, Location::Device(SetDevice().current_device())));
    allocations.insert(a);

    values.insert(std::shared_ptr<Value>(
        new Value(devPtr, size, a->Id(), false /*initialized*/)));
  } else {
    assert(0 && "How did we get here?");
  }
}

void handleCudaFree(Allocations &allocations, Values &values,
                    const CUpti_CallbackData *cbInfo) {
  (void)values;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaFree entry\n");
    auto params = ((cudaFree_v3020_params *)(cbInfo->functionParams));
    auto devPtr = (uintptr_t)params->devPtr;

    // Find the live matching allocation
    bool found;
    Allocations::key_type allocId;
    std::tie(found, allocId) = allocations.find_live(
        devPtr, Location::Device(SetDevice().current_device()));
    if (found) { // found
    } else {
    }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

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

void handleCudaConfigureCall(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaConfigureCall entry\n");

    if (ConfiguredCall().valid) {
      printf("ERROR: Call is already configured?\n");
    }

    auto params = ((cudaConfigureCall_v3020_params *)(cbInfo->functionParams));
    ConfiguredCall().gridDim = params->gridDim;
    ConfiguredCall().blockDim = params->blockDim;
    ConfiguredCall().sharedMem = params->sharedMem;
    ConfiguredCall().stream = params->stream;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

void handleCudaSetupArgument(const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaSetupArgument entry\n");
    const auto params =
        ((cudaSetupArgument_v3020_params *)(cbInfo->functionParams));
    const uintptr_t arg =
        (uintptr_t) * static_cast<const void *const *>(
                          params->arg); // arg is a pointer to the arg.
    // const size_t size     = params->size;
    // const size_t offset   = params->offset;

    ConfiguredCall().args.push_back(arg);
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
}

void handleCudaLaunch(Values &values, const CUpti_CallbackData *cbInfo) {
  lazyStopCallbacks();
  lazyActivateCallbacks();
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaLaunch entry\n");
    // const auto params = ((cudaLaunch_v3020_params
    // *)(cbInfo->functionParams));
    // const uintptr_t func = (uintptr_t) params->func;

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

    // Assume that the kernel can modify each argument value.
    // Therefore, each modifiable argument generates a new value.
    // All of these new values depend on the previous argument values.
    for (const auto &argKey : kernelArgKeys) {
      const auto &argValue = values[argKey];
      const auto newValue =
          std::shared_ptr<Value>(new Value(*argValue)); // duplicate the value
      values.insert(newValue);
      for (const auto &depKey : kernelArgKeys) {
        printf("launch: %lu deps on %lu\n", newValue->Id(), depKey);
        newValue->add_depends_on(depKey);
      }
    }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }

  printf("callback: cudaLaunch: done\n");
}

std::string getCallbackName(CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid) {
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API:
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamDestroy_v3020:
      return std::string("cudaStreamDestroy_v3020");
    default:
      return std::string("<unknown runtime api: ") + std::to_string(cbid) +
             std::string(">");
    }
    break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    switch (cbid) {
    default:
      return std::string("<unknown driver api: ") + std::to_string(cbid) +
             std::string(">");
    }
  }
  default:
    return std::string("<unknown domain>");
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
      handleMemcpy(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleCudaMalloc(Allocations::instance(), Values::instance(), cbInfo);
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
    default:
      auto name = getCallbackName(domain, cbid);
      printf("skipping runtime call %s...\n", name.c_str());
      break;
    }
  } break;
  case CUPTI_CB_DOMAIN_DRIVER_API: {
    auto name = getCallbackName(domain, cbid);
    printf("skipping driver call %s...\n", name.c_str());
    break;
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

void lazyStopCallbacks() {
  if (SUBSCRIBER_ACTIVE) {
    printf("suspending callbacks...\n");
    stopCallbacks();
    SUBSCRIBER_ACTIVE = false;
  }
}

void lazyActivateCallbacks() {
  if (!SUBSCRIBER_ACTIVE) {
    printf("activating callbacks...\n");
    activateCallbacks();
    SUBSCRIBER_ACTIVE = true;
  }
}