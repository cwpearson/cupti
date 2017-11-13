#ifndef VALUES_HPP
#define VALUES_HPP

#include <memory>
#include <mutex>

#include "allocations.hpp"
#include "value.hpp"

// FIXME: implement overlap with interval tree
// could use boost icl with map of positions to std::set<value_type>

class Values {
public:
  typedef Value::id_type id_type;
  typedef std::shared_ptr<Value> value_type;
  static const id_type noid;

private:
  typedef std::map<id_type, value_type> map_type;

  map_type values_;
  std::vector<id_type> value_order_;
  std::mutex access_mutex_;

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

  id_type find_id(const uintptr_t pos, const AddressSpace &as) const;

  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::pair<map_type::iterator, bool> insert(const Value &v);

  std::pair<id_type, value_type> duplicate_value(const value_type &v) {
    auto nv = std::shared_ptr<Value>(new Value(*v));
    auto p = insert(nv);
    assert(p.second && "Should be new value");
    return *p.first;
  }

  std::pair<id_type, value_type> new_value(const uintptr_t pos,
                                           const size_t size,
                                           const Allocations::id_type allocId) {
    return new_value(pos, size, allocId, false);
  }

  std::pair<id_type, value_type> new_value(const uintptr_t pos,
                                           const size_t size,
                                           const Allocations::id_type allocId,
                                           const bool initialized) {
    assert((allocId != noid) && "Allocation should be valid");

    auto v = new Value(pos, size, allocId, initialized);
    auto p = insert(std::shared_ptr<Value>(v));
    assert(p.second && "Expecting new value");
    return *p.first;
  }

  value_type &operator[](const id_type &k) { return values_[k]; }
  value_type &operator[](id_type &&k) { return values_[k]; }

  static Values &instance();

private:
  Values();
  std::string output_path_;
};

#endif