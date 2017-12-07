#ifndef CPROF_VALUE_HPP
#define CPROF_VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "cprof/address_space.hpp"
#include "util/extent.hpp"

namespace cprof {

class Value : public Extent {

private:
  friend class Allocations;
  friend class Allocation;
  AddressSpace address_space_;
  bool initialized_;
  size_t id_;

public:
  Value(size_t id, const uintptr_t pos, const size_t size,
        const AddressSpace &as, const bool initialized)
      : Extent(pos, size), id_(id), address_space_(as),
        initialized_(initialized) {}

  void add_depends_on(const Value &V);
  std::string json() const;

  operator bool() const noexcept { return id_ != 0; }
  size_t id() const { return id_; }
  AddressSpace address_space() const;
  void set_size(const size_t size);
  bool initialized() const { return initialized_; }
};

} // namespace cprof

#endif
