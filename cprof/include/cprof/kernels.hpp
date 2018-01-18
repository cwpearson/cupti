#ifndef CPROF_KERNELS_HPP
#define CPROF_KERNELS_HPP

#include <map>
#include <mutex>
#include <utility>

#include "cprof/model/cuda/kernel_record.hpp"

using cprof::model::cuda::KernelRecord_t;

namespace cprof {
class Kernels {
public:
  typedef uint64_t key_type;
  typedef KernelRecord_t mapped_type;
  typedef std::pair<key_type, mapped_type> value_type;
  typedef std::map<key_type, mapped_type> container_type;
  typedef container_type::iterator iterator;

  std::pair<iterator, bool> insert(const mapped_type &m);
  mapped_type at(const key_type &k);

private:
  container_type records_;
  std::mutex mux_;
};
} // namespace cprof
#endif