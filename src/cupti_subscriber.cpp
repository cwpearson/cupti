#include <iostream>

#include "cprof/callbacks.hpp"
#include "cprof/activity_callbacks.hpp"
#include "cprof/cupti_subscriber.hpp"
#include "cprof/kernel_time.hpp"
#include "cprof/util_cupti.hpp"

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size,
                                     size_t *maxNumRecords) {
  printf("INFO: bufferRequested\n");
  uint8_t *rawBuffer;

  *size = BUFFER_SIZE * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = BUFFER_SIZE;

  if (*buffer == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }
}


  CuptiSubscriber::CuptiSubscriber(CUpti_CallbackFunc callback) : callback_(callback) {
  }

  void CuptiSubscriber::init() {
    assert(callback_);
    printf("INFO: CuptiSubscriber init\n");
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);

    printf("INFO: activating callbacks\n");
    CUPTI_CHECK(
        cuptiSubscribe(&subscriber_, callback_, nullptr));
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
    printf("INFO: done activating callbacks\n");
  }

  CuptiSubscriber::~CuptiSubscriber() {
    printf("INFO: CuptiSubscriber dtor\n");
    cuptiActivityFlushAll(0);
    auto kernelTimer = KernelCallTime::instance();
    kernelTimer.write_to_file();
    kernelTimer.close_parent();
    printf("Deactivating callbacks!\n");
    CUPTI_CHECK(cuptiUnsubscribe(subscriber_));
    printf("INFO: done deactivating callbacks!\n");
  }

