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
#include "value.hpp"

#define CHECK_CU_ERROR(err, cufunc)                                            \
  if (err != CUDA_SUCCESS) {                                                   \
    printf("%s:%d: error %d for CUDA Driver API function '%s'\n", __FILE__,    \
           __LINE__, err, cufunc);                                             \
    exit(-1);                                                                  \
  }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                                      \
  if (err != CUPTI_SUCCESS) {                                                  \
    const char *errstr;                                                        \
    cuptiGetResultString(err, &errstr);                                        \
    printf("%s:%d:Error %s for CUPTI API function '%s'.\n", __FILE__,          \
           __LINE__, errstr, cuptifunc);                                       \
    exit(-1);                                                                  \
  }

typedef uint64_t Time;

Time getTimestamp(const CUpti_CallbackData *cbInfo) {
  uint64_t time;
  CUptiResult cuptiErr = cuptiDeviceGetTimestamp(cbInfo->context, &time);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetTimestamp");
  return Time(time);
}

void handleMemcpy(Allocations &allocations, Values &values,
                  const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {

    // extract API call parameters
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    const uintptr_t dst = (uintptr_t)params->dst;
    const uintptr_t src = (uintptr_t)params->src;
    const cudaMemcpyKind kind = params->kind;
    const size_t count = params->count;

    // Look for existing src / dst allocations
    bool srcFound, dstFound;
    Allocations::key_type srcAlloc, dstAlloc;
    Allocation::Location srcLoc, dstLoc;
    if (cudaMemcpyHostToDevice == kind) {
      printf("%lu --[h2d]--> %lu\n", src, dst);
      srcLoc = Allocation::Location::Host;
      dstLoc = Allocation::Location::Device;
    } else if (cudaMemcpyDeviceToHost == kind) {
      printf("%lu --[d2h]--> %lu\n", src, dst);
      srcLoc = Allocation::Location::Device;
      dstLoc = Allocation::Location::Host;
    } else if (cudaMemcpyDeviceToDevice == kind) {
      srcLoc = Allocation::Location::Device;
      dstLoc = Allocation::Location::Device;
    } else {
      assert(0 && "Unsupported cudaMemcpy kind");
    }

    std::tie(srcFound, srcAlloc) = allocations.find_live(src, count, srcLoc);
    std::tie(dstFound, dstAlloc) = allocations.find_live(dst, count, dstLoc);

    // always creates a new dst value
    auto dstVal = std::shared_ptr<Value>(new Value(dst, count));
    values.insert(dstVal);
    auto dstIdx = dstVal->Id();

    // See if there is a value for the source, or create one
    uintptr_t srcIdx;
    bool found;
    std::tie(found, srcIdx) = values.get_last_overlapping_value(src, count);
    fprintf(stderr, "here\n");
    if (found) {
      values[dstIdx]->depends_on(srcIdx);
      if (!values[srcIdx]->is_known_size()) {
        printf("WARN: source is unknown size. Setting by memcpy count\n");
        values[srcIdx]->set_size(count);
      }
      printf("found existing srcId %lu for %lu\n", srcIdx, src);
    } else {
      auto srcVal = std::shared_ptr<Value>(new Value(src, count));
      values.insert(srcVal);
      auto srcIdx = srcVal->Id();
      values[dstIdx]->depends_on(srcIdx);
    }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    for (auto kv : values) {
      const auto &valIdx = kv.first;
      const auto &val = kv.second;
      if (val->depends_on().size() > 0) {
        printf("%lu <- ", valIdx);
        for (const auto &d : val->depends_on()) {
          printf("%lu ", d);
        }
        printf("\n");
      }
    }
  } else {
    assert(0 && "How did we get here?");
  }
}

void handleMalloc(Allocations &allocations, Values &values,
                  const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
    uintptr_t devPtr = (uintptr_t)(*(params->devPtr));
    const size_t size = params->size;
    printf("[cudaMalloc] %lu[%lu]\n", devPtr, size);

    std::shared_ptr<Allocation> a(
        new Allocation(devPtr, size, Allocation::Location::Device));
    allocations.insert(a);

    values.insert(std::shared_ptr<Value>(new Value(devPtr, size)));
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
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    printf("callback: cudaLaunch entry\n");
    // const auto params = ((cudaLaunch_v3020_params
    // *)(cbInfo->functionParams));
    // const uintptr_t func = (uintptr_t) params->func;

    // Find all values that are used by arguments
    std::vector<size_t> argValIds;
    for (size_t argIdx = 0; argIdx < ConfiguredCall().args.size();
         ++argIdx) { // for each kernel argument
      // printf("arg %lu, val %lu\n", argIdx, valIdx);
      bool found;
      uintptr_t valIdx;
      std::tie(found, valIdx) =
          values.get_last_overlapping_value(ConfiguredCall().args[argIdx], 1);
      if (found) {
        argValIds.push_back(valIdx);
      }
    }

    // create a new value for each argument. All of these depend on all the
    // argument values
    for (size_t i = 0; i < ConfiguredCall().args.size(); ++i) {
      const auto &arg = ConfiguredCall().args[i];
      std::shared_ptr<Value> newValue(new Value(arg, 0));
      for (const auto &id : argValIds) {
        newValue->depends_on(id);
      }
      values.insert(newValue);
    }

  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  } else {
    assert(0 && "How did we get here?");
  }
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
  // uint64_t startTimestamp;
  // uint64_t endTimestamp;
  (void)userdata;

  // Data is collected for the following APIs
  switch (domain) {
  case CUPTI_CB_DOMAIN_RUNTIME_API: {
    switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleMemcpy(Allocations::instance(), Values::instance(), cbInfo);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
      handleMalloc(Allocations::instance(), Values::instance(), cbInfo);
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

static const char *memcpyKindStr(enum cudaMemcpyKind kind) {
  switch (kind) {
  case cudaMemcpyHostToDevice:
    return "HostToDevice";
  case cudaMemcpyDeviceToHost:
    return "DeviceToHost";
  default:
    break;
  }

  return "<unknown>";
}

std::string myexec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
  if (!pipe)
    throw std::runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != NULL)
      result += buffer.data();
  }
  return result;
}

/*
int main(int argc, char **argv) {

  Records records;


  CUcontext context = 0;
  CUdevice device = 0;
  CUresult cuerr;
  CUptiResult cuptierr;


  cuerr = cuInit(0);
  CHECK_CU_ERROR(cuerr, "cuInit");

  cuerr = cuCtxCreate(&context, 0, device);
  CHECK_CU_ERROR(cuerr, "cuCtxCreate");

  CUpti_SubscriberHandle runtimeSubscriber;
  cuptierr = cuptiSubscribe(&runtimeSubscriber,
(CUpti_CallbackFunc)runtimeCallback , &records);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber,
CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber,
CUPTI_CB_DOMAIN_DRIVER_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");

  cudaSetDevice(0);
    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        cmd += std::string(argv[i]) + std::string(" ");
    }
    printf("Executing %s\n", cmd.c_str());
    int status = system(cmd.c_str());
    printf("Done executing %s\n", cmd.c_str());

  cuptierr = cuptiUnsubscribe(runtimeSubscriber);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");


  printf("%lu\n", records.size());
  for (auto &r : records) {
    printf("%d\n", r->id_);
  }
}
*/

int initCallbacks() {

  // CUdevice device = 0;
  // CUresult cuerr;
  CUptiResult cuptierr;

  CUpti_SubscriberHandle runtimeSubscriber;
  cuptierr =
      cuptiSubscribe(&runtimeSubscriber, (CUpti_CallbackFunc)callback, nullptr);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");
  cuptierr =
      cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");
  cuptierr =
      cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");

  // cuptierr = cuptiUnsubscribe(runtimeSubscriber);
  // CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");

  return 0;
}

void lazyInitCallbacks() {
  static int initialized = 0;
  if (!initialized) {
    printf("registering callbacks...\n");
    initCallbacks();
    initialized = 1;
  } else {
  }
  return;
}
