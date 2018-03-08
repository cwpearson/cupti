#ifndef CPROF_VALUE_HPP
#define CPROF_VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cprof/address_space.hpp"
#include "util/extent.hpp"

namespace cprof {

class Value : public Extent {

private:
  friend class Allocations;
  friend class Allocation;
  size_t allocation_;
  AddressSpace addressSpace_;
  bool initialized_;
  size_t id_;
  std::string creationReason_;
  size_t creatorId_;

public:
  Value(size_t id, const uintptr_t pos, const size_t size,
        const size_t &allocation, const AddressSpace &addressSpace,
        const bool initialized, const std::string &creationReason,
        const size_t creatorId)
      : Extent(pos, size), allocation_(allocation), addressSpace_(addressSpace),
        initialized_(initialized), id_(id), creationReason_(creationReason),
        creatorId_(creatorId) {}
  Value(size_t id, const uintptr_t pos, const size_t size,
        const size_t &allocation, const AddressSpace &addressSpace,
        const bool initialized)
      : Value(id, pos, size, allocation, addressSpace, initialized, "", 0) {}
  Value()
      : Value(0, 0, 0, 0, AddressSpace::Unknown(), false, "DEFAULT_VALUE_CTOR",
              0) {}

  void add_depends_on(const Value &V, const uint64_t apiId) const;
  std::string json() const;

  explicit operator bool() const noexcept { return id_ != 0; }
  size_t id() const { return id_; }
  const AddressSpace &address_space() const noexcept;
  void set_size(const size_t size);
  bool initialized() const { return initialized_; }
  bool operator==(const Value &rhs) const;
  bool operator!=(const Value &rhs) const;
  const std::string &creation_reason() const { return creationReason_; }
  const size_t &creator_id() const { return creatorId_; }
  void set_creator_id(const size_t &id) { creatorId_ = id; }
};

} // namespace cprof

std::ostream &operator<<(std::ostream &os, const cprof::Value &dt);

#endif
