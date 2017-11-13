#include "callbacks.hpp"
#include "util_cupti.hpp"

class CuptiSubscriber {
private:
  CUpti_SubscriberHandle subscriber_;

public:
  CuptiSubscriber(CUpti_CallbackFunc callback) {
    printf("Activating callbacks!\n");
    CUPTI_CHECK(
        cuptiSubscribe(&subscriber_, (CUpti_CallbackFunc)callback, nullptr));
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API));
    CUPTI_CHECK(cuptiEnableDomain(1, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  ~CuptiSubscriber() {
    printf("Deactivating callbacks!\n");
    CUPTI_CHECK(cuptiUnsubscribe(subscriber_));
  }
};

// Subscribe and unsubscribe at global scope
CuptiSubscriber Manager((CUpti_CallbackFunc)callback);