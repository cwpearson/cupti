#include <vector>
#include <cassert>

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

Time getTimestamp(const CUpti_CallbackData *cbInfo) {
  uint64_t time;
  CUptiResult cuptiErr = cuptiDeviceGetTimestamp(cbInfo->context, &time);
  CHECK_CUPTI_ERROR(cuptiErr, "cuptiDeviceGetTimestamp");
  return Time(time);
}

void handleMemcpy(Records &records, const CUpti_CallbackData *cbInfo) {

  MemcpyRecord *record = nullptr;

  // Create a new record on entrance, or look up an existing record on exit
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    record = new MemcpyRecord();

    record->bytes_ = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->count;
    record->kind_  = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->kind;
        
    record->start_ = getTimestamp(cbInfo);
    } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      // Find the existing record
      for (auto r : records) {
        if (cbInfo->correlationId == r->id_) {
          record = static_cast<MemcpyRecord*>(r);
        }
      }
      assert(record);
    record->end_ = getTimestamp(cbInfo);
    } else {
      assert(0 && "How did we get here?");
    }
}



void CUPTIAPI
getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  static int memTransCount = 0;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  auto records = reinterpret_cast<Records*>(userdata);
  CUptiResult cuptiErr;
      
  // Data is collected for the following APIs
  switch (cbid) {
    CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      handleMemcpy(*records, cbInfo);
      break;
    default:
      printf("Skipping...\n");
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


int main(int argc, char **argv) {

  Records records;


  CUcontext context = 0;
  CUdevice device = 0;
  CUresult cuerr;
  CUptiResult cuptierr;

  CUpti_SubscriberHandle subscriber;
    
  cuerr = cuInit(0);
  CHECK_CU_ERROR(cuerr, "cuInit");

  cuerr = cuCtxCreate(&context, 0, device);
  CHECK_CU_ERROR(cuerr, "cuCtxCreate");

  cuptierr = cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback , &records);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiSubscribe");

  cuptierr = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiEnableDomain");


    std::string cmd;
    for (int i = 1; i < argc; ++i) {
        cmd += std::string(argv[i]) + std::string(" ");
    }
    printf("Executing %s\n", cmd.c_str());
    int status = system(cmd.c_str());
    printf("Done executing %s\n", cmd.c_str());

  cuptierr = cuptiUnsubscribe(subscriber);
  CHECK_CUPTI_ERROR(cuptierr, "cuptiUnsubscribe");
}
