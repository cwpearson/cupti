#ifndef CPROF_MODEL_DRIVER_STATE_HPP
#define CPROF_MODEL_DRIVER_STATE_HPP

#include <map>
#include <mutex>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cupti.h>

#include "cprof/api_record.hpp"
#include "cprof/model/cuda/configured_call.hpp"
#include "cprof/model/thread.hpp"
#include "util/logging.hpp"

namespace cprof {
namespace model {

class ThreadState {
private:
  int currentDevice_;
  bool cuptiCallbacksEnabled_;

  ConfiguredCall configuredCall_;
  std::vector<ApiRecordRef> apiStack_;

public:
  ThreadState() : currentDevice_(0), cuptiCallbacksEnabled_(true) {}

  int current_device() const { return currentDevice_; }
  void set_device(const int device) { currentDevice_ = device; }

  void api_enter(const int device, const CUpti_CallbackDomain domain,
                 const CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);
  void api_exit(const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
                const CUpti_CallbackData *cbInfo);

  bool in_child_api() const { return apiStack_.size() >= 2; }
  const ApiRecordRef &parent_api() const;
  ApiRecordRef &current_api();

  void pause_cupti_callbacks();
  void resume_cupti_callbacks();

  bool is_cupti_callbacks_enabled() const { return cuptiCallbacksEnabled_; }

  ConfiguredCall &configured_call() { return configuredCall_; }
};

// FIXME: not thread-safe
class Driver {
public:
  typedef ThreadState mapped_type;
  typedef tid_t key_type;
  typedef std::pair<key_type, mapped_type> value_type;

private:
  typedef std::map<key_type, mapped_type> ThreadMap;
  ThreadMap threadStates_;
  std::map<const cublasHandle_t, int> cublasHandleToDevice_;
  std::map<const cudnnHandle_t, int> cudnnHandleToDevice_;
  std::mutex access_mutex_;

public:
  void track_cublas_handle(const cublasHandle_t h, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    cublasHandleToDevice_[h] = device;
  }
  void track_cudnn_handle(const cudnnHandle_t h, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    // logging::err() << "DEBU: tracking cudnn handle " << h << std::endl;
    cudnnHandleToDevice_[h] = device;
    // logging::err() << cudnnHandleToDevice_.size() << std::endl;
  }
  int device_from_cublas_handle(const cublasHandle_t h) {
    logging::err() << "DEBU: looking for cublas handle " << h << std::endl;
    return cublasHandleToDevice_.at(h);
  }

  int device_from_cudnn_handle(const cudnnHandle_t h) {
    logging::err() << "DEBU: looking for cudnn handle " << h << " "
                   << cudnnHandleToDevice_.size() << std::endl;

    for (const auto kv : cudnnHandleToDevice_) {
      logging::err() << kv.first << "," << kv.second << std::endl;
    }

    return cudnnHandleToDevice_.at(h);
  }
  mapped_type &this_thread() { return threadStates_[get_thread_id()]; }
};

} // namespace model
} // namespace cprof

#endif
