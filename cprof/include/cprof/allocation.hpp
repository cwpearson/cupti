#ifndef CPROF_ALLOCATION_HPP
#define CPROF_ALLOCATION_HPP

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
#include "cprof/value.hpp"
#include "util/interval_set.hpp"
#include "util/logging.hpp"

class AllocationRecord {
private:
  static size_t next_val_;
  size_t val_;
  bool val_initialized_;

public:
  uintptr_t lower_;
  uintptr_t upper_;
  AddressSpace address_space_;
  cprof::model::Memory memory_;
  cprof::model::tid_t thread_id_;
  cprof::model::Location location_;
  bool freed_;

  typedef uintptr_t pos_type;
  pos_type lower() const noexcept { return lower_; }
  pos_type upper() const noexcept { return upper_; }
  void set_lower(const pos_type &p) { lower_ = p; }
  void set_upper(const pos_type &p) { upper_ = p; }

  AllocationRecord(uintptr_t pos, size_t size, const AddressSpace &as,
                   const cprof::model::Memory &mem,
                   const cprof::model::Location &location)
      : val_(next_val_++), val_initialized_(0), lower_(pos), upper_(pos + size),
        address_space_(as), memory_(mem), location_(location), freed_(false) {}
  AllocationRecord(const uintptr_t pos, const size_t size)
      : AllocationRecord(pos, size, AddressSpace::Host(),
                         cprof::model::Memory::Unknown,
                         cprof::model::Location::Unknown()) {}
  AllocationRecord() : AllocationRecord(0, 0) {}

  std::string json() const;

  cprof::Value new_value(uintptr_t pos, size_t size, const bool initialized);
  cprof::Value value(const uintptr_t pos, const size_t size) const;

  pos_type pos() const noexcept { return lower_; }
  const AddressSpace &address_space() const { return address_space_; }
  void free() { freed_ = true; }
};

class Allocation {

private:
  std::shared_ptr<AllocationRecord> ar_;

public: // interval interface
  typedef uintptr_t pos_type;
  pos_type lower() const noexcept { return ar_->lower_; }
  pos_type upper() const noexcept { return ar_->upper_; }
  void set_lower(const pos_type &p) { ar_->lower_ = p; }
  void set_upper(const pos_type &p) { ar_->upper_ = p; }

public:
  Allocation(AllocationRecord *ar)
      : ar_(std::shared_ptr<AllocationRecord>(ar)) {}
  Allocation(std::shared_ptr<AllocationRecord> p) : ar_(p) {}
  Allocation() : Allocation(nullptr) {}

  std::string json() const;

  pos_type pos() const noexcept;
  size_t size() const noexcept;
  AddressSpace address_space() const;
  cprof::model::Memory memory() const;
  cprof::model::Location location() const;
  bool freed() const noexcept;
  size_t id() const noexcept;
  void free();

  bool operator!() const noexcept;
  explicit operator bool() const noexcept;
  bool operator==(const Allocation &rhs) const noexcept;

  cprof::Value new_value(uintptr_t pos, size_t size, const bool initialized) {
    return ar_->new_value(pos, size, initialized);
  }
  cprof::Value value(const uintptr_t pos, const size_t size) const {
    return ar_->value(pos, size);
  }
};
#endif
