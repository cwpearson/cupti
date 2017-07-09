#ifndef DRIVER_STATE_HPP
#define DRIVER_STATE_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>

#include "thread.hpp"

class ThreadState {
private:
  std::map<tid_t, ThreadState> threadState_;
  int currentDevice_;

public:
  int current_device() const { return currentDevice_; }
  void set_device(const int device) { currentDevice_ = device; }
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
