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

class Allocation {
private:
  static size_t next_val_;
  size_t val_;
  bool val_initialized_;

private:
  static size_t nextId_;

private:
  AddressSpace address_space_;
  cprof::model::Memory memory_;
  cprof::model::Location location_;
  bool freed_;
  size_t id_;

public:
  typedef uintptr_t pos_type;
  pos_type lower_;
  pos_type upper_;
  cprof::model::tid_t thread_id_;

public: // interval interface
  pos_type lower() const noexcept { return lower_; }
  pos_type upper() const noexcept { return upper_; }
  void set_lower(const pos_type &p) { lower_ = p; }
  void set_upper(const pos_type &p) { upper_ = p; }

public:
  Allocation(pos_type pos, size_t size, const AddressSpace &as,
             const cprof::model::Memory &mem,
             const cprof::model::Location &location)
      : val_(0), id_(nextId_), lower_(pos), upper_(pos + size),
        address_space_(as), memory_(mem), location_(location), freed_(false) {
    ++nextId_;
  }
  Allocation(const pos_type &pos, const size_t &size)
      : Allocation(pos, size, AddressSpace::Host(),
                   cprof::model::Memory::Unknown,
                   cprof::model::Location::Unknown()) {}
  Allocation() : Allocation(0, 0) {}

  std::string json() const;

  cprof::Value new_value(pos_type pos, size_t size, const bool initialized) {
    assert(pos >= lower_ && pos + size < upper_);
    if (!val_) {
      val_ = next_val_++;
      val_initialized_ = initialized;
    }
    return value(pos, size);
  }
  cprof::Value value(const uintptr_t pos, const size_t size) const {
    return cprof::Value(val_, pos, size, address_space_, val_initialized_);
  }

  pos_type pos() const noexcept;
  size_t size() const noexcept;
  AddressSpace address_space() const;
  cprof::model::Memory memory() const;
  cprof::model::Location location() const;
  bool freed() const noexcept;
  size_t id() const noexcept;
  void free();

  bool operator!() const noexcept { return pos() == 0; }
  explicit operator bool() const noexcept { return pos() != 0; }
  bool operator==(const Allocation &rhs) const noexcept {
    return pos() == rhs.pos() && size() == rhs.size() &&
           address_space() == rhs.address_space();
  }
};
#endif
