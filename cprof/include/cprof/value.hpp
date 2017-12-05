#ifndef CPROF_VALUE_HPP
#define CPROF_VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "allocation.hpp"
#include "util/extent.hpp"

namespace cprof {

class ValueRecord : public Extent {
  friend std::ostream &operator<<(std::ostream &os, const ValueRecord &v);

public:
  typedef int64_t id_type;

private:
  friend class Values;
  friend class Value;
  Allocation allocation_;
  bool initialized_;

  ValueRecord(const Extent::pos_t pos, const size_t size,
              const Allocation &alloc, const bool initialized)
      : Extent(pos, size), allocation_(alloc), initialized_(initialized) {}

public:
  ValueRecord &operator+=(const ValueRecord &rhs) {
    *this = rhs;
    return *this;
  }

  void add_depends_on(const ValueRecord &VR);
  std::string json() const;

  AddressSpace address_space() const;
  void set_size(const size_t size);
  bool initialized() const { return initialized_; }
  const Allocation &allocation() const { return allocation_; }
};

/*! \brief Thin wrapper to a ValueRecord
 *
 * Value is a std::shared_ptr<ValueRecord> with additional operations for
 * use with boost::icl::interval_map
 */
class Value {
  friend std::ostream &operator<<(std::ostream &os, const ValueRecord &v);

public:
  Value(ValueRecord *p) : p_(std::shared_ptr<ValueRecord>(p)) {}
  Value() : Value(nullptr) {}

private:
  std::shared_ptr<ValueRecord> p_;

public:
  /*! \brief Needed for icl
   *
   *  Used by boost::icl::interval_map for combining Value when a new
   * interval/value pair is inserted. This effectively has the inserted one
   * overwrite the old one on that interval.
   */
  Value &operator+=(const Value &rhs) {
    *this = rhs;
    return *this;
  }

  ValueRecord &operator*() const noexcept { return p_.operator*(); }
  ValueRecord *operator->() const noexcept { return p_.operator->(); }
  /*! \brief Needed for icl
   */
  bool operator==(const Value &rhs) const noexcept { return p_ == rhs.p_; }
  explicit operator bool() const noexcept { return bool(p_); }
  void add_depends_on(const Value &other) { return p_->add_depends_on(*other); }
  ValueRecord *get() const noexcept { return p_.get(); }
};
std::ostream &operator<<(std::ostream &os, const Value &v);
/*
class Value : public Extent {
public:
  typedef uintptr_t id_type;
  static const id_type noid;

private:
  bool is_initialized_;
  Allocation allocation_;

public:
  friend std::ostream &operator<<(std::ostream &os, const Value &v);

  void add_depends_on(id_type id);
  const std::vector<size_t> &depends_on() const { return dependsOnIdx_; }
  bool is_known_size() const { return size_ != 0; }

  AddressSpace address_space() const;
  std::string json() const;
  void set_size(size_t size);

  id_type Id() const { return reinterpret_cast<id_type>(this); }

  Value(uintptr_t pos, size_t size, const Allocation &allocation,
        bool initialized)
      : Extent(pos, size), is_initialized_(initialized),
        allocation_(allocation) {}

  Value(uintptr_t pos, size_t size, const Allocation &allocation)
      : Value(pos, size, allocation, false) {}

  void record_meta_append(const std::string &s);
  void record_meta_set(const std::string &s);

private:
  std::vector<id_type> dependsOnIdx_; // values this value depends on
};
*/

} // namespace cprof

#endif
