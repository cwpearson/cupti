#ifndef API_RECORD_HPP
#define API_RECORD_HPP

#include <vector>

#include "values.hpp"

class ApiRecord {
public:
  typedef uintptr_t id_type;
  static const id_type noid;

private:
  std::vector<Values::id_type> inputs_;
  std::vector<Values::id_type> outputs_;
  std::string name_;
  int device_;

public:
  friend std::ostream &operator<<(std::ostream &os, const ApiRecord &r);

  ApiRecord(const std::string &name, const int device)
      : name_(name), device_(device) {}

  void add_input(const Value::id_type &id);
  void add_output(const Value::id_type &id);

  int device() const { return device_; }
  id_type Id() const { return reinterpret_cast<id_type>(this); }
  const std::string &name() const { return name_; }

  std::string json() const;
};

#endif