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
  std::string apiName_;
  std::string kernelName_;
  int device_;
  uint64_t start_;
  uint64_t end_;

public:
  friend std::ostream &operator<<(std::ostream &os, const ApiRecord &r);

  ApiRecord(const std::string &name, const int device)
      : apiName_(name), device_(device), start_(0), end_(0) {}
  ApiRecord(const std::string &apiName, const std::string &kernelName,
            const int device)
      : ApiRecord(apiName, device) {
    kernelName_ = kernelName;
  }

  void add_input(const Value::id_type &id);
  void add_output(const Value::id_type &id);

  void record_start_time(const uint64_t start);
  void record_end_time(const uint64_t end);

  int device() const { return device_; }
  id_type Id() const { return reinterpret_cast<id_type>(this); }
  const std::string &name() const { return apiName_; }

  std::string json() const;
};

#endif