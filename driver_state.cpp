#include "driver_state.hpp"

DriverState &DriverState::instance() {
  static DriverState s;
  return s;
}
