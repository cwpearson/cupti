#ifndef VALUE_HPP
#define VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "allocation_record.hpp"
#include "extent.hpp"

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

#endif
