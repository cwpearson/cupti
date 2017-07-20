#include "driver_state.hpp"
#include "util_cupti.hpp"

#include <cassert>

DriverState &DriverState::instance() {
  static DriverState s;
  return s;
}

void ThreadState::api_enter(const CUpti_CallbackDomain domain,
                            const CUpti_CallbackId cbid,
                            const CUpti_CallbackData *cbInfo) {

  apiStack_.push_back(APICallRecord(domain, cbid, cbInfo));
  // printf("Entering %s [stack sz=%lu]\n", cbInfo->functionName,
  //        apiStack_.size());
}

void ThreadState::api_exit(const CUpti_CallbackDomain domain,
                           const CUpti_CallbackId cbid,
                           const CUpti_CallbackData *cbInfo) {

  // printf("Exiting %s [stack sz=%lu]\n", cbInfo->functionName,
  // apiStack_.size());
  assert(!apiStack_.empty());
  const APICallRecord current = apiStack_.back();
  assert(current.domain() == domain);
  assert(current.cbid() == cbid);
  assert(current.cb_info() == cbInfo);
  apiStack_.pop_back();
}

const APICallRecord &ThreadState::parent_api() const {
  assert(apiStack_.size() >= 2);
  return apiStack_[apiStack_.size() - 2];
}

APICallRecord &ThreadState::current_api() {
  assert(apiStack_.size() >= 1);
  return apiStack_.back();
}

void ThreadState::pause_cupti_callbacks() {
  assert(cuptiCallbacksEnabled_);
  cuptiCallbacksEnabled_ = false;
}
void ThreadState::resume_cupti_callbacks() {
  assert(!cuptiCallbacksEnabled_);
  cuptiCallbacksEnabled_ = true;
}