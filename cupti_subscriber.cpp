#include "callbacks.hpp"
#include "activity_callbacks.hpp"
#include "util_cupti.hpp"
#include <iostream>

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

  static void CUPTIAPI
  bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
  {
    uint8_t *rawBuffer;
  
    *size = 100000 * 1024;
    rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);
  
    *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
    *maxNumRecords = 100000;
  
    if (*buffer == NULL) {
      printf("Error: out of memory\n");
      exit(-1);
    }
  }

class CuptiSubscriber {
private:
  CUpti_SubscriberHandle subscriber_;

public:
  CuptiSubscriber(CUpti_CallbackFunc callback) {
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  
    printf("Activating callbacks!\n");
    CUPTI_CHECK(
        cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)callback, nullptr));
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  ~CuptiSubscriber() {
    auto kernelTimer = KernelCallTime::instance();
    kernelTimer.write_to_file();
    cuptiActivityFlushAll(0);
    printf("Deactivating callbacks!\n");
    CUPTI_CHECK(cuptiUnsubscribe(subscriber_));
  }
};

// Subscribe and unsubscribe at global scope
CuptiSubscriber Manager((CUpti_CallbackFunc)callback);