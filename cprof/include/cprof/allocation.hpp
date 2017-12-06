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

class Allocation {

public:
  typedef uint64_t id_type;

private:
  uintptr_t pos_;
  size_t size_;
  AddressSpace address_space_;
  cprof::model::Memory memory_;
  cprof::model::tid_t thread_id_;
  cprof::model::Location location_;
  id_type id_;
  bool freed_;

  static id_type unique_id() {
    static id_type count = 0;
    return count++;
  }

public:
  Allocation(uintptr_t pos, size_t size, const AddressSpace &as,
             const cprof::model::Memory &mem,
             const cprof::model::Location &location, size_t id)
      : pos_(pos), size_(size), address_space_(as), memory_(mem),
        location_(location), id_(id), freed_(false) {}
  Allocation(uintptr_t pos, size_t size, const AddressSpace &as,
             const cprof::model::Memory &mem,
             const cprof::model::Location &location)
      : Allocation(pos, size, as, mem, location, unique_id()) {
    assert(address_space_.is_valid());
  }
  Allocation(const uintptr_t pos, const size_t size)
      : Allocation(pos, size, AddressSpace::Host(),
                   cprof::model::Memory::Unknown,
                   cprof::model::Location::Unknown()) {}
  Allocation() : Allocation(0, 0) {}

  std::string json() const;

  id_type id() const { return id_; }
  AddressSpace address_space() const { return address_space_; }
  cprof::model::Memory memory() const { return memory_; }
  cprof::model::Location location() const { return location_; }

  bool freed() const noexcept { return freed_; }
  uintptr_t pos() const noexcept { return pos_; }
  size_t size() const noexcept { return size_; }

  boost::icl::interval<uintptr_t>::interval_type interval() const {
    return boost::icl::interval<uintptr_t>::right_open(pos_, pos_ + size_);
  }

  explicit operator bool() const noexcept { return pos_; }

  Allocation &operator+=(const Allocation &rhs) {
    logging::err() << "adding " << pos_ << " +" << size_ << " to " << rhs.pos_
                   << " +" << rhs.size_ << "\n";

    if (freed_) {
      *this = rhs;

    } else {

      // Merge the allocations
      auto overlapStart = std::min(pos_, rhs.pos_);
      auto overlapEnd = std::max(pos_ + size_, rhs.pos_ + rhs.size_);

      pos_ = overlapStart;
      size_ = overlapEnd - overlapStart;
    }
    return *this;
  }
  bool operator==(const Allocation &rhs) const noexcept {
    if (pos_ == 0 && rhs.pos_ == 0) {
      return true;
    }
    return pos_ == rhs.pos_ && size_ == rhs.size_ &&
           address_space_ == rhs.address_space_;
  }
};

/*
namespace boost {
namespace icl {

template <> struct interval_traits<Allocation> {

  typedef Allocation interval_type;
  typedef uintptr_t domain_type;
  typedef std::less<uintptr_t> domain_compare;
  static interval_type construct(const domain_type &lo, const domain_type &up) {
    logging::err() << "construct called\n";
    return interval_type(lo, up - lo);
  }
  // 3.2 Selection of values
  static domain_type lower(const interval_type &inter_val) {
    return inter_val.pos();
  };
  static domain_type upper(const interval_type &inter_val) {
    return inter_val.pos() + inter_val.size();
  };
};

template <>
struct interval_bound_type<Allocation> // 4.  Finally we define the interval
                                       // borders.
{ //    Choose between static_open         (lo..up)
  typedef interval_bound_type
      type; //                   static_left_open    (lo..up]
  BOOST_STATIC_CONSTANT(bound_type, value = interval_bounds::static_right_open);
}; //               and static_closed       [lo..up]

} // namespace icl
} // namespace boost
*/
#endif
