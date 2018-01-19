#include <cassert>

#include "cprof/model/driver.hpp"
#include "cprof/util_cupti.hpp"

using namespace cprof::model;

void ThreadState::api_enter(const int device, const CUpti_CallbackDomain domain,
                            const CUpti_CallbackId cbid,
                            const CUpti_CallbackData *cbInfo) {

  apiStack_.push_back(
      std::make_shared<ApiRecord>(device, domain, cbid, cbInfo));
}

void ThreadState::api_exit(const CUpti_CallbackDomain domain,
                           const CUpti_CallbackId cbid,
                           const CUpti_CallbackData *cbInfo) {

  assert(!apiStack_.empty());
  const auto current = apiStack_.back();
  assert(current->domain() == domain);
  assert(current->cbid() == cbid);
  assert(current->cb_info() == cbInfo);
  apiStack_.pop_back();
}

const ApiRecordRef &ThreadState::parent_api() const {
  assert(apiStack_.size() >= 2);
  return apiStack_[apiStack_.size() - 2];
}

ApiRecordRef &ThreadState::current_api() {
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