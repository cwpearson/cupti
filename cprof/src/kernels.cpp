#include "cprof/kernels.hpp"

using cprof::Kernels;

std::pair<Kernels::iterator, bool> Kernels::insert(const mapped_type &m) {
  std::lock_guard<std::mutex> access(mux_);
  auto key = records_.size();
  return records_.insert(std::make_pair(key, m));
}

Kernels::mapped_type Kernels::at(const key_type &k) {
  std::lock_guard<std::mutex> access(mux_);
  return records_.at(k);
}
