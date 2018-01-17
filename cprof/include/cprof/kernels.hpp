#ifndef CPROF_KERNELS_HPP
#define CPROF_KERNELS_HPP

#include <map>
#include <utility>

#include "cprof/model/cuda/kernel_record.hpp"

using cprof::model::cuda::KernelRecord;

namespace cprof {
class Kernels {
public:
  typedef uint64_t key_type;
  typedef KernelRecord mapped_type;
  typedef std::pair<key_type, mapped_type> value_type;
  typedef std::map<key_type, mapped_type> container_type;
  typedef container_type::iterator iterator;

  std::pair<iterator, bool> insert(const mapped_type &m) {
    auto key = records_.size();
    return records_.insert(std::make_pair(key, m));
  }

private:
  container_type records_;
};
} // namespace cprof
#endif