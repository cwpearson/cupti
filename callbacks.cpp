#include <vector>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cupti.h>

#include "data.hpp"

#define CHECK_CU_ERROR(err, cufunc)                                     \
  if (err != CUDA_SUCCESS)                                              \
    {                                                                   \
      printf ("%s:%d: error %d for CUDA Driver API function '%s'\n",    \
              __FILE__, __LINE__, err, cufunc);                         \
      exit(-1);                                                         \
    }

#define CHECK_CUPTI_ERROR(err, cuptifunc)                               \
  if (err != CUPTI_SUCCESS)                                             \
    {                                                                   \
      const char *errstr;                                               \
      cuptiGetResultString(err, &errstr);                               \
      printf ("%s:%d:Error %s for CUPTI API function '%s'.\n",          \
              __FILE__, __LINE__, errstr, cuptifunc);                   \
      exit(-1);                                                         \
    }


typedef uint64_t Time;

Time getTimestamp(const CUpti_CallbackData *cbInfo) {
  uint64_t time;
  CUptiResult cuptiErr = cuptiDeviceGetTimestamp(cbInfo->context, &time);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetTimestamp");
  return Time(time);
}

void handleMemcpy(Allocations &allocations, Values &values, const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    const uintptr_t dst = (uintptr_t) params->dst;
    const uintptr_t src = (uintptr_t) params->src;
    const cudaMemcpyKind kind = params->kind;
      //const size_t count = params->count;


    if (cudaMemcpyHostToDevice == kind) {
      printf("%lu --[h2d]--> %lu\n", src, dst);

      // h2d always creates a new dst value
      auto dstIdx = values.size();
      values.push_back(Value());
      
      // See if there is a value for the source, or create one
      bool found = false;
      for (size_t i = 0; i < values.size(); ++i) {
        if (src == values[i].pos_) {
          values[dstIdx].dependsOnIdx_.push_back(i);
          found = true;
        }
      }
      if (!found) {
        size_t srcIdx = values.size();
        values.push_back(Value());
        values[srcIdx].pos_ = src;
        values[dstIdx].dependsOnIdx_.push_back(srcIdx);
      }
    } else if (cudaMemcpyDeviceToHost == kind) {
      printf("%lu --[d2h]--> %lu\n", src, dst);

      // h2d always creates a new dst value
      auto dstIdx = values.size();
      values.push_back(Value());
      
      // See if there is a value for the source, or create one
      bool found = false;
      for (size_t i = 0; i < values.size(); ++i) {
        if (src == values[i].pos_) {
          values[dstIdx].dependsOnIdx_.push_back(i);
          found = true;
        }
      }
      if (!found) {
        size_t srcIdx = values.size();
        values.push_back(Value());
        values[srcIdx].pos_ = src;
        values[dstIdx].dependsOnIdx_.push_back(srcIdx);
      }
    }
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    for (size_t i = 0; i < values.size(); ++i) {
      auto &v = values[i];
      for (const auto &d : v.dependsOnIdx_) {
        printf("%lu <- %lu\n", i, d);
      }
    }
  } else {
    assert(0 && "How did we get here?");
  }

}

void handleMalloc(Allocations &allocations, Values &values, const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
  void **devPtr = params->devPtr;
  const size_t size = params->size;
  printf("[malloc] %lu[%lu]\n", (uintptr_t)(*devPtr), size);

  Allocation a;
  a.pos_ = (uintptr_t) *devPtr;
  a.size_ = size;
  allocations.push_back(a);

  Value newValue;
  newValue.allocationIdx_ = allocations.size() - 1;
  newValue.pos_ = (uintptr_t) *devPtr;
  values.push_back(newValue);
  } else {
    assert(0 && "How did we get here?");
  }
}

void CUPTIAPI
callback(void *userdata, CUpti_CallbackDomain domain,
         CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  auto &data = *(reinterpret_cast<Data*>(userdata));
  CUptiResult cuptiErr;
      
  // Data is collected for the following APIs
  switch (domain) {
    case CUPTI_CB_DOMAIN_RUNTIME_API:
      switch (cbid) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
          handleMemcpy(data.allocations_, data.values_,  cbInfo);
          break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
          handleMalloc(data.allocations_, data.values_, cbInfo);
          break;
        default:
          break;
      }
      break;
    case CUPTI_CB_DOMAIN_DRIVER_API:
      //printf("Callback domain: driver\n");
      break;
    default:
      //printf("Callback domain: other\n");
      break;
  }
}

static const char *
memcpyKindStr(enum cudaMemcpyKind kind)
{
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

std::string myexec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");
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
  cuptierr = cuptiSubscribe(&runtimeSubscriber, (CUpti_CallbackFunc)runtimeCallback , &records);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_DRIVER_API);
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

  CUdevice device = 0;
  CUresult cuerr;
  CUptiResult cuptierr;

  CUpti_SubscriberHandle runtimeSubscriber;
  cuptierr = cuptiSubscribe(&runtimeSubscriber, (CUpti_CallbackFunc)callback , &Data::instance());
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");

  //cuptierr = cuptiUnsubscribe(runtimeSubscriber);
  //CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");

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

