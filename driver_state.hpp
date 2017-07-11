#ifndef DRIVER_STATE_HPP
#define DRIVER_STATE_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <cassert>
#include <map>
#include <utility>
#include <vector>

#include "thread.hpp"

class APICallRecord {
private:
  CUpti_CallbackDomain domain_;
  CUpti_CallbackId cbid_;
  const CUpti_CallbackData *cbInfo_;

public:
  bool is_runtime() const { return domain_ == CUPTI_CB_DOMAIN_RUNTIME_API; }
  CUpti_CallbackId cbid() const { return cbid_; }
  const CUpti_CallbackData *cb_info() const { return cbInfo_; }
  const CUpti_CallbackDomain domain() const { return domain_; }
  APICallRecord()
      : domain_(CUPTI_CB_DOMAIN_INVALID), cbid_(-1), cbInfo_(nullptr) {}
  APICallRecord(const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
                const CUpti_CallbackData *cbInfo)
      : domain_(domain), cbid_(cbid), cbInfo_(cbInfo) {}
};

class ThreadState {
private:
  std::map<tid_t, ThreadState> threadState_;
  int currentDevice_;

  std::vector<APICallRecord> apiStack_;

public:
  int current_device() const { return currentDevice_; }
  void set_device(const int device) { currentDevice_ = device; }

  void api_enter(const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
                 const CUpti_CallbackData *cbInfo) {
    apiStack_.push_back(APICallRecord(domain, cbid, cbInfo));
  }
  void api_exit(const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
                const CUpti_CallbackData *cbInfo) {

    assert(!apiStack_.empty());
    const APICallRecord current = apiStack_.back();
    assert(current.domain() == domain);
    assert(current.cbid() == cbid);
    assert(current.cb_info() == cbInfo);
    apiStack_.pop_back();
  }
  bool in_child_api() const { return apiStack_.size() >= 2; }
  const APICallRecord &parent_api() const {
    assert(apiStack_.size() >= 2);
    return apiStack_[apiStack_.size() - 2];
  }
};

class DriverState {
private:
  typedef std::map<tid_t, ThreadState> ThreadMap;
  ThreadMap threadState_;

  static DriverState &instance();

public:
  static ThreadState &thread(const tid_t &tid) {
    return instance().threadState_[tid];
  }
};

#endif
