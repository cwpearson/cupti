#ifndef SET_DEVICE_HPP
#define SET_DEVICE_HPP

typedef struct {
  int device_ = -1;
  int current_device() const { return device_; }
} SetDevice_t;

SetDevice_t &SetDevice();

#endif