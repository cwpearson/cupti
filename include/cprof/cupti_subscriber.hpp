#ifndef CPROF_CUPTI_SUBSCRIBER_HPP
#define CPROF_CUPTI_SUBSCRIBER_HPP

class CuptiSubscriber {
private:
  CUpti_SubscriberHandle subscriber_;
  CUpti_CallbackFunc callback_;

public:
  CuptiSubscriber(CUpti_CallbackFunc callback);
  void init();
  ~CuptiSubscriber();
};

#endif