#ifndef VALUE_HPP
#define VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "allocation.hpp"
#include "extent.hpp"

const std::string output_path("cprof.txt");

class Value : public Extent {
private:
  bool is_unknown_;
  bool is_not_value_;
  Allocation::id_type allocation_id_; // allocation that this value lives in

public:
  typedef uintptr_t id_type;
  friend std::ostream &operator<<(std::ostream &os, const Value &v);

  void add_depends_on(id_type id);
  const std::vector<size_t> &depends_on() const { return dependsOnIdx_; }
  bool is_known_size() const { return size_ != 0; }

  id_type Id() const { return reinterpret_cast<id_type>(this); }
  Location location() const;
  std::string json() const;
  void set_size(size_t size);

  Value(uintptr_t pos, size_t size, Allocation::id_type allocation)
      : Extent(pos, size), is_unknown_(false), is_not_value_(false),
        allocation_id_(allocation) {}

private:
  static Value &UnknownValue();
  static Value &NoValue();
  std::vector<id_type> dependsOnIdx_; // values this value depends on
};

class Values {
public:
  typedef std::shared_ptr<Value> value_type;

private:
  typedef uintptr_t key_type;
  typedef std::map<key_type, value_type> map_type;
  map_type values_;
  std::vector<key_type> value_order_;

public:
  std::pair<bool, key_type>
  get_last_overlapping_value(uintptr_t pos, size_t size, Location loc);
  value_type find_live(uintptr_t pos, size_t size, Location loc);
  value_type find_live(uintptr_t pos, Location loc);

  std::pair<map_type::iterator, bool> insert(const value_type &v);

  value_type &operator[](const key_type &k) { return values_[k]; }
  value_type &operator[](key_type &&k) { return values_[k]; }
  map_type::iterator begin() { return values_.begin(); }
  map_type::iterator end() { return values_.end(); }

  static Values &instance();

private:
  Values();
  std::string output_path_;
};

#endif
