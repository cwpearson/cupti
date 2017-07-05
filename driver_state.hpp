#ifndef DRIVER_STATE_HPP
#define DRIVER_STATE_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <map>

class DriverState {
private:
  std::map<cudaStream_t, int> streamToDevice_;
  int currentDevice_;

  int _current_device() const {return currentDevice_;}
  void _set_device(const int device) {currentDevice_ = device;}
  int _stream_device(const cudaStream_t stream) { return streamToDevice_.at(stream); }
  void _create_stream(const cudaStream_t stream) { streamToDevice_[stream] = _current_device(); }

  DriverState() : currentDevice_(0) {}
  static DriverState &instance();

public:
  static int current_device() { return instance()._current_device(); }
  static void set_device(const int device) { instance()._set_device(device); }
  static int stream_device(const cudaStream_t stream) { return instance()._stream_device(stream); }
  static void create_stream(const cudaStream_t stream) { instance()._create_stream(stream); }
};

#endif
