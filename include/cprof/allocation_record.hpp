#ifndef ALLOCATIONRECORD_HPP
#define ALLOCATIONRECORD_HPP

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>

#include "address_space.hpp"
#include "cprof/model/location.hpp"
#include "cprof/model/memory.hpp"
#include "cprof/model/thread.hpp"
#include "util/extent.hpp"

class AllocationRecord : public Extent {
public:
  typedef uintptr_t id_type;
  static const id_type noid;

private:
  AddressSpace address_space_;
  cprof::model::Memory memory_;
  cprof::model::tid_t thread_id_;
  cprof::model::Location location_;
  bool freed_;

public:
  AllocationRecord(uintptr_t pos, size_t size, const AddressSpace &as,
                   const cprof::model::Memory &mem,
                   const cprof::model::Location &location);

  std::string json() const;

  bool overlaps(const AllocationRecord &other) {
    return (address_space_.maybe_equal(other.address_space_)) &&
           Extent::overlaps(other);
  }

  bool contains(uintptr_t pos, size_t size, const AddressSpace &as) {
    return address_space_ == as && Extent::contains(Extent(pos, size));
  }

  bool contains(uintptr_t pos, size_t size) {
    return Extent::contains(Extent(pos, size));
  }

  bool contains(const AllocationRecord &other) {
    return (address_space_.maybe_equal(other.address_space_)) &&
           Extent::contains(other);
  }

  id_type Id() const { return reinterpret_cast<id_type>(this); }
  AddressSpace address_space() const { return address_space_; }
  cprof::model::Memory memory() const { return memory_; }

  void mark_free() { freed_ = true; }
  bool freed() const { return freed_; }
};

typedef std::shared_ptr<AllocationRecord> Allocation;

#endif
