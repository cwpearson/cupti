#ifndef ALLOCATIONRECORD_HPP
#define ALLOCATIONRECORD_HPP

#include <cassert>
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
  friend class Allocation;

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

class Allocation {
private:
  typedef AllocationRecord element_type;

public:
  Allocation(element_type *p) : p_(std::shared_ptr<element_type>(p)) {}
  Allocation() : Allocation(nullptr) {}

private:
  std::shared_ptr<element_type> p_;

public:
  /*! \brief Fuse allocations when they overlap
   */
  Allocation &operator+=(Allocation &rhs) {
    // update us to also cover rhs
    const auto myStart = p_->pos();
    const auto myEnd = p_->pos() + p_->size();
    const auto rhsStart = rhs->pos();
    const auto rhsEnd = rhsStart + rhs->size();
    assert((myStart >= rhsStart && myStart < rhsEnd) ||
           (myEnd > rhsStart && myEnd <= rhsEnd)); // should overlap
    assert(p_->address_space() == rhs->address_space());

    const uintptr_t overlapStart = std::min(myStart, rhsStart);
    const uintptr_t overlapEnd = std::max(myEnd, rhsEnd);
    const size_t overlapSize = overlapEnd - overlapStart;
    p_->pos_ = overlapStart;
    p_->size_ = overlapSize;

    // point rhs at this so there are duplicates that we can clean up later
    rhs = *this;

    return *this;
  }

  AllocationRecord &operator*() const noexcept { return p_.operator*(); }
  AllocationRecord *operator->() const noexcept { return p_.operator->(); }
  /*! \brief Needed for icl
   */
  bool operator==(const Allocation &rhs) const noexcept { return p_ == rhs.p_; }
  explicit operator bool() const noexcept { return bool(p_); }
  AllocationRecord *get() const noexcept { return p_.get(); }
};

#endif
