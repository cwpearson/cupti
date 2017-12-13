//
// Created by pearson on 12/8/17.
//

#ifndef INTERVAL_CONTAINER_INTERVAL_SET_HPP
#define INTERVAL_CONTAINER_INTERVAL_SET_HPP

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

typedef int Direction;

// INTERVAL must have ::pos_type, type lower(), type upper(), bool operator<()
// When intervals overlap, they are joined
template <typename INTERVAL> class IntervalSet {
  typedef INTERVAL key_type;
  typedef size_t size_type;
  typedef std::shared_ptr<INTERVAL> ptr_type;
  typedef std::pair<typename INTERVAL::pos_type, Direction> Endpoint;
  typedef std::map<Endpoint, ptr_type> map_type;

  class iterator {
    friend class IntervalSet;

  public:
    iterator(const typename map_type::iterator lb,
             const typename map_type::iterator &ub)
        : lb_(lb), ub_(ub) {}
    const ptr_type &operator*() { // FIXME - const?
      return lb_->second;
    }
    const typename map_type::iterator operator->() { return lb_; }
    bool operator==(const iterator &rhs) const noexcept {
      return lb_ == rhs.lb_ && ub_ == rhs.ub_;
    }
    bool operator!=(const iterator &rhs) const noexcept {
      return !operator==(rhs);
    }

    iterator &operator++() {
      lb_++;
      lb_++;
      ub_++;
      ub_++;
      return *this;
    }

  private:
    typename map_type::iterator lb_;
    typename map_type::iterator ub_;
  };

private:
  map_type map_;

  static Endpoint make_lower(const typename INTERVAL::pos_type &i) {
    return Endpoint(i, 1);
  }

  static Endpoint make_upper(const typename INTERVAL::pos_type &i) {
    return Endpoint(i, -1);
  }

  // returns the largest element lte e
  typename map_type::iterator lte(const Endpoint &e) {
    auto lb = map_.lower_bound(e); // lb >= e
    if (lb == map_.end()) {
      if (!map_.empty()) { // there's something in the container, but it's
                           // smaller than e. return it
        --lb;
        return lb;
      }
      return map_.end();
    }
    if (lb->first == e) {
      return lb;
    } else { // greater than e. return the first thing smaller than e
      --lb;
      return lb;
    }
  }

  // returns the largest element lt e
  typename map_type::iterator lt(const Endpoint &e) {
    auto lb = map_.lower_bound(e); // lb >= e
    if (lb == map_.end()) {
      if (!map_.empty()) { // there's something in the container, but it's
                           // smaller than e. return it
        --lb;
        return lb;
      }
      return map_.end();
    }
    --lb;
    return lb;
  }

  iterator find(const Endpoint &e) {
    auto lteI = lte(e);
    auto gtI = map_.upper_bound(e);

    if (lteI == map_.end()) {
      return end();
    }
    if (gtI == map_.end()) {
      return end();
    }
    if (lteI->first.second == 1 && gtI->first.second == -1) {
      return iterator(lteI, gtI);
    } else {
      return end();
    }
  }

  iterator find_between(const Endpoint &l, const Endpoint &u) {
    auto ltI = lte(u);
    auto gteI = map_.lower_bound(l);

    if (ltI == map_.end()) {
      return end();
    }
    if (gteI == map_.end()) {
      return end();
    }

    if (gteI->first < u && gteI->first.second == 1 && ltI->first.second == -1) {
      std::cerr << "here\n";
      return iterator(gteI, ltI);
    } else {
      return end();
    }
  }

  // return if lower is <= e and upper end is > e
  typename map_type::iterator enclosed_by_lower(const Endpoint &e) {

    std::cerr << "Looking for lower of " << e.first << "\n";

    auto gteE = map_.lower_bound(e);
    if (gteE == map_.end()) {
      std::cerr << "no lb\n";
      return gteE;
    }
    if (gteE == map_.begin()) {
      std::cerr << "lb is first map element\n";
      return gteE;
    }

    auto ltE = gteE;
    --ltE;
    std::cerr << "found candidate " << ltE->first.first << "\n";
    if (ltE->first.second == -1) { // points away from e
      std::cerr << "candidate points in wrong dir " << ltE->first.first << "\n";
      return map_.end();
    }
    if (gteE == e) {
      std::cerr << "upper bound eqal " << gteE->first.first << "\n";
      return map_.end();
    }

    return ltE;
  }

public:
  size_type size() const {
    assert(map_.size() % 2 == 0);
    return map_.size() / 2;
  }

  std::pair<iterator, bool> insert_join(const key_type &k) {

    std::cerr << "(insert_join) current map:\n";
    int cnt = 0;
    for (const auto &i : map_) {
      std::cerr << i.first.first << "(" << i.first.second << ") ";
      if (cnt++ % 2)
        std::cerr << "\n";
    }

    const auto kUpperEnd = make_upper(k.upper());
    const auto kLowerEnd = make_lower(k.lower());

    auto small =
        map_.insert(std::make_pair(kLowerEnd, ptr_type(nullptr))).first;
    auto large =
        map_.insert(std::make_pair(kUpperEnd, ptr_type(nullptr))).first;

    // std::cerr << "inserted\n";

    // find smallest endpoint facing +1 lte kLowerEnd
    if (small != map_.begin()) {
      auto smaller = small;
      smaller--;
      if (smaller->first.second == 1) {
        small = smaller;
      }
    }

    // std::cerr << "small = " << small->first.first << "\n";

    // find the largest endpoint facing -1 gte kUpperEnd
    auto larger = large;
    larger++;
    if (larger != map_.end()) {
      if (larger->first.second == -1) {
        large = larger;
      }
    }

    // std::cerr << "large = " << large->first.first << "\n";
    bool inserted = false;

    // Adjust the values of the new interval to be that of the overlapped
    INTERVAL newK = k;
    newK.set_lower(small->first.first);
    newK.set_upper(large->first.first);

    // update those endpoints to point at the new interval
    if (small->second) {
      *(small->second) = newK;
      large->second = small->second;
    } else if (large->second) {
      *(large->second) = newK;
      small->second = large->second;
    } else {
      auto newRecord = ptr_type(new key_type(newK));
      large->second = newRecord;
      small->second = newRecord;
      inserted = true;
    }

    iterator res(small, large);

    // erase all endpoints between
    std::vector<typename map_type::iterator> toErase;
    while (++small != large) {
      toErase.push_back(small);
    }

    for (const auto &i : toErase) {
      // std::cerr << "erasing " << i->first.first << "\n";
      map_.erase(i);
    }
    assert(map_.size() % 2 == 0);

    return std::make_pair(res, inserted);
  }

  std::pair<iterator, bool> insert_split(const key_type &k) {

    // std::cerr << "(insert_split) current map:\n";
    // int cnt = 0;
    // for (const auto &i : map_) {
    //   std::cerr << i.first.first << "(" << i.first.second << ") ";
    //   if (cnt++ % 2)
    //     std::cerr << "\n";
    // }

    const auto kUpperEnd = make_upper(k.upper());
    const auto kLowerEnd = make_lower(k.lower());

    auto newK = ptr_type(new key_type(k));

    auto p = map_.insert(std::make_pair(kLowerEnd, newK));
    auto newL = p.first;
    if (!p.second) { // not inserted
      newL->second->set_lower(k.lower());
      newL->second->set_upper(k.upper());
    }
    p = map_.insert(std::make_pair(kUpperEnd, newK));
    auto newU = p.first;
    if (!p.second) { // not inserted
      newU->second->set_lower(k.lower());
      newU->second->set_upper(k.upper());
    }

    bool inserted = false;

    // find largest endpoint facing +1 lt kLowerEnd
    auto smaller = newL;
    if (smaller != map_.begin()) {
      smaller--;
      if (smaller->first.second == 1) { // truncate existing interval
        auto oldP = smaller->second;
        ptr_type newP = ptr_type(new key_type(*oldP));
        newP->set_upper(k.lower());
        inserted = true;
        // std::cerr << "patching " << smaller->second->lower() << " "
        //           << smaller->second->upper() << "\n";
        smaller->second = newP;
        map_.insert(std::make_pair(make_upper(k.lower()), smaller->second));
        // std::cerr << "into " << smaller->second->lower() << " "
        //           << smaller->second->upper() << "\n";
      }
    }

    // find smallest endpoint facing -1 gt kUpperEnd
    auto bigger = map_.upper_bound(kUpperEnd);
    if (bigger != map_.end()) {
      if (bigger->first.second == -1) { // truncate existing interval
        inserted = true;
        auto oldP = bigger->second;
        ptr_type newP = ptr_type(new key_type(*oldP));
        newP->set_lower(k.upper());
        std::cerr << "patching " << bigger->second->lower() << " "
                  << bigger->second->upper() << "\n";
        bigger->second = newP;
        map_.insert(std::make_pair(make_lower(k.upper()), bigger->second));
        std::cerr << "into " << bigger->second->lower() << " "
                  << bigger->second->upper() << "\n";
      }
    }

    iterator res(newL, newU);

    // erase all endpoints between the new insertion
    std::vector<typename map_type::iterator> toErase;
    while (++newL != newU) {
      toErase.push_back(newL);
    }

    for (const auto &i : toErase) {
      std::cerr << "erasing " << i->first.first << "\n";
      map_.erase(i);
    }
    assert(map_.size() % 2 == 0);

    return std::make_pair(res, inserted);
  }

  iterator end() { return iterator(map_.end(), map_.end()); }

  iterator find(const typename INTERVAL::pos_type &p) {
    return find(make_lower(p));
  }

  iterator find(const typename INTERVAL::pos_type &pos, const size_t size) {
    const auto kLowerEnd = make_lower(pos);
    const auto kUpperEnd = make_upper(pos + size);
    return find(kLowerEnd, kUpperEnd);
  }

  iterator find(const Endpoint &l, const Endpoint &u) {
    auto lowerInt = find(l); // interval containing lower end
    if (lowerInt != end()) {
      return lowerInt;
    }
    auto upperInt = find(u); // interval containing upper end
    if (upperInt != end()) {
      return upperInt;
    }
    return find_between(l, u);
  }

  iterator find(const key_type &k) {

    std::cerr << "(find) current map:\n";
    int cnt = 0;
    for (const auto &i : map_) {
      std::cerr << i.first.first << "(" << i.first.second << ") ";
      if (cnt++ % 2)
        std::cerr << "\n";
    }

    // Convert to endpoints and defer
    assert(map_.size() % 2 == 0);
    const auto kUpperEnd = make_upper(k.upper());
    const auto kLowerEnd = make_lower(k.lower());
    return find(kLowerEnd, kUpperEnd);
  }

  size_type erase(const key_type &k) {
    auto i = find(k);
    if (i != end()) {
      map_.erase(i.lb_);
      map_.erase(i.ub_);
      return 1;
    }
    return 0;
  }

  static Endpoint lower_endpoint(int64_t i) { return Endpoint(i, 1); }
  static Endpoint upper_endpoint(int64_t i) { return Endpoint(i, -1); }
};

#endif // INTERVAL_CONTAINER_INTERVAL_SET_HPP
