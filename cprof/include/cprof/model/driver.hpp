#ifndef CPROF_MODEL_DRIVER_STATE_HPP
#define CPROF_MODEL_DRIVER_STATE_HPP

#include <cassert>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cupti.h>
#include <nccl.h>

#include "cprof/api_record.hpp"
#include "cprof/model/cuda/configured_call.hpp"
#include "cprof/model/location.hpp"
#include "cprof/model/thread.hpp"
#include "cprof/util_numa.hpp"
#include "util/logging.hpp"

namespace cprof {
namespace model {

class ThreadState {
private:
  int currentDevice_;
  bool cuptiCallbacksEnabled_;
  std::vector<CUcontext> contextStack_;

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

  void push_context(const CUcontext c) { contextStack_.push_back(c); }
  void pop_context() {
    assert(!contextStack_.empty());
    contextStack_.resize(contextStack_.size() - 1);
  }
  void set_context(const CUcontext c) {
    if (c == 0) {
      if (!contextStack_.empty()) {
        pop_context();
      }
    } else if (!contextStack_.empty()) {
      contextStack_[contextStack_.size() - 1] = c;
    } else {
      push_context(c); // FIXME: not clear if this is right from docs
    }
  }
  CUcontext current_context() const {
    assert(!contextStack_.empty());
    return contextStack_.back();
  }

  bool is_cupti_callbacks_enabled() const { return cuptiCallbacksEnabled_; }

  Location pause_cupti_get_location(const void *p) {

/*
    cudaPointerAttributes attrs;
    pause_cupti_callbacks();
    cudaError_t ret = cudaPointerGetAttributes(&attrs, p);
    resume_cupti_callbacks();
    if (ret != cudaSuccess) {
      return Location::Unknown();
    }

    if (attrs.memoryType == cudaMemoryTypeHost) {
      int numaNode = get_numa_node(p);
      return Location::Host(numaNode);
    } else if (attrs.memoryType == cudaMemoryTypeDevice) {
      return Location::CudaDevice(attrs.device);
    } else {
      assert(0 && "Unexpected cudaMemoryType");
    }
*/
    return Location::Unknown();
    assert(0 && "How did we get here?");
  }
  Location pause_cupti_get_location(const uintptr_t p) {
    return pause_cupti_get_location(reinterpret_cast<const void *>(p));
  }

  ConfiguredCall &configured_call() { return configuredCall_; }
};

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
  std::map<const ncclComm_t, int> ncclCommToDevice_;
  std::mutex access_mutex_;

public:
  void track_cublas_handle(const cublasHandle_t h, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    cublasHandleToDevice_[h] = device;
  }
  void track_cudnn_handle(const cudnnHandle_t h, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    cudnnHandleToDevice_[h] = device;
  }
  void register_ncclComm(const ncclComm_t c, const int device) {
    std::lock_guard<std::mutex> guard(access_mutex_);
    ncclCommToDevice_[c] = device;
  }

  int device_from_cublas_handle(const cublasHandle_t h) {
    std::stringstream ss;
    ss << "DEBU: looking for cublas handle " << h << std::endl;
    logging::atomic_err(ss.str());
    return cublasHandleToDevice_.at(h);
  }

  int device_from_cudnn_handle(const cudnnHandle_t h) {
    std::stringstream ss;
    ss << "DEBU: looking for cudnn handle " << h << std::endl;
    logging::atomic_err(ss.str());
    return cudnnHandleToDevice_.at(h);
  }

  int device(const ncclComm_t c) {
    std::stringstream ss;
    ss << "DEBU: looking for nccl comm " << c << std::endl;
    logging::atomic_err(ss.str());
    return ncclCommToDevice_.at(c);
  }

  mapped_type &this_thread() { return threadStates_[get_thread_id()]; }
};

} // namespace model
} // namespace cprof

#endif
