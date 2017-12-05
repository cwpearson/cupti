#ifndef VALUES_HPP
#define VALUES_HPP

#include <memory>
#include <mutex>

#include "allocations.hpp"
#include "value.hpp"

#include <boost/icl/interval_map.hpp>

using boost::icl::interval;
using boost::icl::interval_map;
namespace cprof {

class Values {
private:
  interval_map<uintptr_t, Value> values_;
  std::mutex access_mutex_;

public:
  Value new_value(const uintptr_t pos, const size_t size,
                  const Allocation &alloc, const bool initialized);

  Value find_live(uintptr_t pos, size_t size, const AddressSpace &as);
  Value find_live(const uintptr_t pos, const AddressSpace &as) {
    return find_live(pos, 1, as);
  }

  Value duplicate_value(const Value &v) {
    // ensure the existing value exists
    auto orig = find_live(v->pos(), v->size(), v->address_space());
    assert(orig);
    return new_value(v->pos(), v->size(), v->allocation(), v->initialized());
  }
};

} // namespace cprof

/*
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

  std::pair<Values::id_type, Values::value_type>
  find_live(uintptr_t pos, const AddressSpace &as) {
    return find_live(pos, 1, as);
  }

  id_type find_id(const uintptr_t pos, const AddressSpace &as) const;

  std::pair<map_type::iterator, bool> insert(const value_type &v);
  std::pair<map_type::iterator, bool> insert(const Value &v);

  std::pair<id_type, value_type> duplicate_value(const value_type &v) {
    auto nv = std::shared_ptr<Value>(new Value(*v));
    auto p = insert(nv);
    assert(p.second && "Should be new value");
    return *p.first;
  }

  std::pair<id_type, value_type>
  new_value(const uintptr_t pos, const size_t size, const Allocation alloc) {
    return new_value(pos, size, alloc, false);
  }

  std::pair<id_type, value_type> new_value(const uintptr_t pos,
                                           const size_t size,
                                           const Allocation alloc,
                                           const bool initialized) {
    assert(alloc.get() && "Allocation should be valid");

    auto v = new Value(pos, size, alloc, initialized);
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
*/

#endif