#ifndef CPROF_VALUE_HPP
#define CPROF_VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "allocation_record.hpp"
#include "util/extent.hpp"

/*
class VB : public Extent {
public:
  VB &operator+=(const VB &rhs) {
    *this = rhs;
    return *this;
  }

  void add_depends_on(const Value &v);
};
*/

class ValueRecord;

class Value {
  private:
  std::shared_ptr<ValueRecord> p_;

  public:
    Value(ValueRecord *p) {
      p_ = std::shared_ptr<ValueRecord>(p);
    }
    Value() : Value(nullptr) {}

    Value &operator+=(const Value &rhs) {
    *this = rhs;
    return *this;
  }

    ValueRecord *operator->() const noexcept {
      return p_.operator->();
  }



  operator bool() const {
    return bool(p_);
  }

  ValueRecord* get() const noexcept {
    return p_.get();
  }
};

class ValueRecord : public Extent {
public:
  typedef int64_t id_type;
private:
  friend class Values;
    friend std::ostream &operator<<(std::ostream &os, const ValueRecord &v);
  id_type id_;
  Allocation allocation_;
  bool initialized_;

public:
  ValueRecord &operator+=(const ValueRecord &rhs) {
    *this = rhs;
    return *this;
  }

  bool operator==(const ValueRecord &rhs) const {
    return (id_ == rhs.id_);
  } 

  ValueRecord() : Extent(0, 0), id_(-1), initialized_(false) {}

  explicit operator bool() const { return id_ >= 0; }

    void add_depends_on(const Value &v);
    std::string json() const;

  AddressSpace address_space() const;  
  void set_size(const size_t size);
  bool initialized() const { return initialized_; }
    const Allocation &allocation() const { return allocation_; }
};



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



#endif
