#ifndef API_RECORD_HPP
#define API_RECORD_HPP

#include <cupti.h>
#include <vector>

#include "cprof/value.hpp"

class ApiRecord {
public:
  typedef uintptr_t id_type;
  static const id_type noid;

private:
  std::vector<cprof::Value> inputs_;
  std::vector<cprof::Value> outputs_;
  std::string apiName_;
  std::string kernelName_;
  int device_;
  uint64_t start_;
  uint64_t end_;
  std::map<std::string, std::string> kv_;

  CUpti_CallbackDomain domain_;
  CUpti_CallbackId cbid_;
  const CUpti_CallbackData *cbInfo_;

public:
  ApiRecord(const std::string &name, const int device)
      : apiName_(name), device_(device), start_(0), end_(0),
        domain_(CUPTI_CB_DOMAIN_INVALID), cbid_(-1), cbInfo_(nullptr) {}
  ApiRecord(const std::string &apiName, const std::string &kernelName,
            const int device)
      : ApiRecord(apiName, device) {
    kernelName_ = kernelName;
  }
  ApiRecord(const int device, const CUpti_CallbackDomain domain,
            const CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
      : apiName_(cbInfo->functionName), device_(device), start_(0), end_(0),
        domain_(domain), cbid_(cbid), cbInfo_(cbInfo) {}

  void add_input(const cprof::Value &v);
  void add_output(const cprof::Value &v);
  void add_kv(const std::string &key, const std::string &val);
  void add_kv(const std::string &key, const size_t &val);

  void record_start_time(const uint64_t start);
  void record_end_time(const uint64_t end);

  int device() const { return device_; }
  id_type Id() const { return reinterpret_cast<id_type>(this); }
  const std::string &name() const { return apiName_; }

  std::string json() const;

  bool is_runtime() const { return domain_ == CUPTI_CB_DOMAIN_RUNTIME_API; }
  CUpti_CallbackDomain domain() const { return domain_; }
  CUpti_CallbackId cbid() const { return cbid_; }
  const CUpti_CallbackData *cb_info() const { return cbInfo_; }
};

typedef std::shared_ptr<ApiRecord> ApiRecordRef;

#endif