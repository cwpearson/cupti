#ifndef DRIVER_STATE_HPP
#define DRIVER_STATE_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <map>
#include <vector>

#include <cublas_v2.h>
#include <cudnn.h>

#include "api_record.hpp"
#include "thread.hpp"

class ThreadState {
private:
  int currentDevice_;
  bool cuptiCallbacksEnabled_;

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
};

// FIXME: not thread-safe
class DriverState {
public:
  typedef ThreadState mapped_type;
  typedef tid_t key_type;
  typedef std::pair<key_type, mapped_type> value_type;

private:
  typedef std::map<key_type, mapped_type> ThreadMap;
  ThreadMap threadStates_;
  std::map<const cublasHandle_t, int> cublasHandleToDevice_;
  std::map<const cudnnHandle_t, int> cudnnHandleToDevice_;

  static DriverState &instance();

public:
  static void track_cublas_handle(const cublasHandle_t h, const int device) {
    instance().cublasHandleToDevice_[h] = device;
  }
  static void track_cudnn_handle(const cudnnHandle_t h, const int device) {
    instance().cudnnHandleToDevice_[h] = device;
  }
  static int device_from_cublas_handle(const cublasHandle_t h) {
    return instance().cublasHandleToDevice_.at(h);
  }

  static int device_from_cudnn_handle(const cudnnHandle_t h) {
    return instance().cudnnHandleToDevice_.at(h);
  }
  static mapped_type &this_thread() {
    return instance().threadStates_[get_thread_id()];
  }
};

#endif
