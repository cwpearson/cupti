#ifndef ALLOCATIONRECORD_HPP
#define ALLOCATIONRECORD_HPP

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>

#include <boost/icl/interval_set.hpp>

#include "address_space.hpp"
#include "cprof/model/location.hpp"
#include "cprof/model/memory.hpp"
#include "cprof/model/thread.hpp"
#include "util/logging.hpp"

using namespace boost::icl;

class AllocationRecord {

public:
  uintptr_t pos_;
  size_t size_;
  AddressSpace address_space_;
  cprof::model::Memory memory_;
  cprof::model::tid_t thread_id_;
  cprof::model::Location location_;
  bool freed_;

public:
  AllocationRecord(uintptr_t pos, size_t size, const AddressSpace &as,
                   const cprof::model::Memory &mem,
                   const cprof::model::Location &location)
      : pos_(pos), size_(size), address_space_(as), memory_(mem),
        location_(location), freed_(false) {}
  AllocationRecord(const uintptr_t pos, const size_t size)
      : AllocationRecord(pos, size, AddressSpace::Host(),
                         cprof::model::Memory::Unknown,
                         cprof::model::Location::Unknown()) {}
  AllocationRecord() : AllocationRecord(0, 0) {}

  std::string json() const;

  boost::icl::interval<uintptr_t>::interval_type interval() const {
    return boost::icl::interval<uintptr_t>::right_open(pos_, pos_ + size_);
  }
};

class Allocation {

public:
private:
  std::shared_ptr<AllocationRecord> ar_;

public:
  Allocation(AllocationRecord *ar)
      : ar_(std::shared_ptr<AllocationRecord>(ar)) {}
  Allocation() : Allocation(nullptr) {}

  std::string json() const { return ar_->json(); }

  uintptr_t pos() const noexcept;
  size_t size() const noexcept;
  AddressSpace address_space() const;
  cprof::model::Memory memory() const;
  cprof::model::Location location() const;
  bool freed() const noexcept;
  void free();
  explicit operator bool() const noexcept { return bool(ar_); }
  bool operator!() const noexcept { return !bool(); }
  uintptr_t id() const { return uintptr_t(ar_.get()); }

  boost::icl::interval<uintptr_t>::interval_type interval() const {
    return ar_->interval();
  }

  // update our underlying interval the incoming one
  Allocation &operator+=(const Allocation &rhs) {
    assert(*this);
    assert(rhs);
    logging::err() << "adding " << pos() << " +" << size() << " to "
                   << rhs.pos() << " +" << rhs.size() << "\n";

    if (freed()) {
      *this = rhs;
    } else {

      // Merge the allocations
      auto overlapStart = std::min(pos(), rhs.pos());
      auto overlapEnd = std::max(pos() + size(), rhs.pos() + rhs.size());

      ar_->pos_ = overlapStart;
      ar_->size_ = overlapEnd - overlapStart;
    }
    return *this;
  }

  bool operator==(const Allocation &rhs) const;
};

#endif
