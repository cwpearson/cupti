#ifndef VALUES_HPP
#define VALUES_HPP

#include <memory>
#include <mutex>

#include "value.hpp"

class Values {
public:
  typedef Value::id_type key_type;
  typedef std::shared_ptr<Value> value_type;

private:
  typedef std::map<key_type, value_type> map_type;
  map_type values_;
  std::vector<key_type> value_order_;
  std::mutex modify_mutex_;

public:
  std::pair<bool, key_type>
  get_last_overlapping_value(uintptr_t pos, size_t size, Location loc);
  std::pair<key_type, value_type> find_live(uintptr_t pos, size_t size,
                                            Location loc);
  std::pair<key_type, value_type> find_live(uintptr_t pos, Location loc);
  std::pair<key_type, value_type> find_live_device(const uintptr_t pos,
                                                   const size_t size);

  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::pair<map_type::iterator, bool> insert(const Value &v);

  value_type &operator[](const key_type &k) { return values_[k]; }
  value_type &operator[](key_type &&k) { return values_[k]; }
  // map_type::iterator begin() { return values_.begin(); }
  // map_type::iterator end() { return values_.end(); }

  static Values &instance();

private:
  Values();
  std::string output_path_;
};

#endif