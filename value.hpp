#ifndef VALUE_HPP
#define VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "allocation_record.hpp"
#include "extent.hpp"

const std::string output_path("cprof.txt");

class Value : public Extent {
public:
  typedef uintptr_t id_type;
  static const id_type noid;

private:
  bool is_initialized_;
  AllocationRecord::id_type
      allocation_id_; // allocation that this value lives in
  std::string label_;

public:
  friend std::ostream &operator<<(std::ostream &os, const Value &v);

  void add_depends_on(id_type id);
  const std::vector<size_t> &depends_on() const { return dependsOnIdx_; }
  bool is_known_size() const { return size_ != 0; }

  AddressSpace address_space() const;
  std::string json() const;
  void set_size(size_t size);

  id_type Id() const { return reinterpret_cast<id_type>(this); }

  Value(uintptr_t pos, size_t size, AllocationRecord::id_type allocation)
      : Extent(pos, size), is_initialized_(false), allocation_id_(allocation) {}

  Value(uintptr_t pos, size_t size, AllocationRecord::id_type allocation,
        bool initialized)
      : Extent(pos, size), is_initialized_(initialized),
        allocation_id_(allocation) {}

  void append_label(const std::string &s);

private:
  // static Value &UnknownValue();
  // static Value &NoValue();
  std::vector<id_type> dependsOnIdx_; // values this value depends on
};

#endif
