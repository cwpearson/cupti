#include "set_device.hpp"

SetDevice_t &SetDevice() {
  static SetDevice_t sd_;
  return sd_;
}
