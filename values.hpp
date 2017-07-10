#ifndef VALUES_HPP
#define VALUES_HPP

#include <memory>
#include <mutex>

#include "value.hpp"

class Values {
public:
  typedef Value::id_type id_type;
  typedef std::shared_ptr<Value> value_type;
  static constexpr id_type noid = Value::noid;

private:
  typedef std::map<id_type, value_type> map_type;
  map_type values_;
  std::vector<id_type> value_order_;
  std::mutex modify_mutex_;

public:
  std::pair<bool, id_type> get_last_overlapping_value(uintptr_t pos,
                                                      size_t size,
                                                      const AddressSpace &as);
  std::pair<id_type, value_type> find_live(uintptr_t pos, size_t size,
                                           const AddressSpace &as);
  std::pair<id_type, value_type> find_live(uintptr_t pos,
                                           const AddressSpace &as);
  std::pair<id_type, value_type> find_live_device(const uintptr_t pos,
                                                  const size_t size);

  id_type find_id(const uintptr_t pos, const AddressSpace &as,
                  const Memory &mem) const;

  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::pair<map_type::iterator, bool> insert(const Value &v);

  std::pair<id_type, value_type> new_value(const uintptr_t pos,
                                           const size_t size);

  value_type &operator[](const id_type &k) { return values_[k]; }
  value_type &operator[](id_type &&k) { return values_[k]; }
  // map_type::iterator begin() { return values_.begin(); }
  // map_type::iterator end() { return values_.end(); }

  static Values &instance();

private:
  Values();
  std::string output_path_;
};

#endif