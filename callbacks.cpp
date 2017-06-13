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




class Record {
 public:
  uint32_t id_; // correlation id
};

class MemcpyRecord : public Record {
 public:
  uint64_t start_;
  uint64_t end_;
  uint64_t bytes_;
  enum cudaMemcpyKind kind_;
};



typedef std::vector<Record*> Records;
typedef uint64_t Time;


class Allocation {
 public:
  int type_;
  uintptr_t pos_;
  size_t size_;
};


class Value {
 public:
  uintptr_t pos_;
  size_t size_;
};

typedef std::vector<Allocation> Allocations;
typedef std::vector<Value> Values;

typedef struct {
  Values values;
  Allocations allocations;
} Data;

static Data data;

Time getTimestamp(const CUpti_CallbackData *cbInfo) {
  uint64_t time;
  CUptiResult cuptiErr = cuptiDeviceGetTimestamp(cbInfo->context, &time);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetTimestamp");
  return Time(time);
}

void handleMemcpy(Allocations &allocations, const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    auto params = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams));
    const void *dst = params->dst;
    const void *src = params->src;
    //const size_t count = params->count;
    //const cudaMemcpyKind kind = params->kind;
    printf("%p -> %p\n", src, dst);
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    } else {
      assert(0 && "How did we get here?");
    }
}

void handleMalloc(Allocations &allocations, const CUpti_CallbackData *cbInfo) {
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
  auto params = ((cudaMalloc_v3020_params *)(cbInfo->functionParams));
  void **devPtr = params->devPtr;
  const size_t size = params->size;
  printf("[malloc] %p[%lu]\n", *devPtr, size);

  Allocation a;
  a.pos_ = (uintptr_t) *devPtr;
  a.size_ = size;
  allocations.push_back(a);

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
          handleMemcpy(data.allocations, cbInfo);
          break;
        case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
          handleMalloc(data.allocations, cbInfo);
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
  cuptierr = cuptiSubscribe(&runtimeSubscriber, (CUpti_CallbackFunc)callback , &data);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");
  cuptierr = cuptiEnableDomain(1, runtimeSubscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");

  //cuptierr = cuptiUnsubscribe(runtimeSubscriber);
  //CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");
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

