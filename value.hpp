#ifndef VALUE_HPP
#define VALUE_HPP

#include <iostream>
#include <map>
#include <memory>
#include <vector>

#include "extent.hpp"

const std::string output_path("cprof.txt");

class Value : public Extent {
private:
  size_t allocation_id_; // allocation that this value lives in
public:
  friend std::ostream &operator<<(std::ostream &os, const Value &v);
  typedef uintptr_t id_type;

  void depends_on(size_t id);
  const std::vector<size_t> &depends_on() const { return dependsOnIdx_; }
  bool is_known_size() const { return size_ != 0; }

  id_type Id() const { return reinterpret_cast<id_type>(this); }
  std::string json() const;
  void set_size(size_t size);

  Value(uintptr_t pos, size_t size) : Extent(pos, size) {}

private:
  std::vector<id_type> dependsOnIdx_; // values this value depends on
};

class Values {
private:
  typedef std::shared_ptr<Value> value_type;
  typedef uintptr_t key_type;
  typedef std::map<key_type, value_type> map_type;
  map_type values_;
  std::vector<key_type> value_order_;

public:
  std::pair<bool, key_type> get_last_overlapping_value(uintptr_t pos,
                                                       size_t size);
  std::pair<map_type::iterator, bool> insert(const value_type &v);

  value_type &operator[](const key_type &k) { return values_[k]; }
  value_type &operator[](key_type &&k) { return values_[k]; }
  map_type::iterator begin() { return values_.begin(); }
  map_type::iterator end() { return values_.end(); }

  static Values &instance();

private:
  Values() : values_(map_type()), value_order_(std::vector<key_type>()) {}
  std::string output_path_;
};

#endif
