#ifndef API_RECORD_HPP
#define API_RECORD_HPP

#include <atomic>
#include <cupti.h>
#include <vector>

#include "cprof/time.hpp"
#include "cprof/value.hpp"

class ApiRecord {
public:
  typedef uint64_t id_type;

private:
  std::vector<cprof::Value> inputs_;
  std::vector<cprof::Value> outputs_;
  std::string apiName_;
  std::string kernelName_;
  int device_;
  std::map<std::string, std::string> kv_;

  CUpti_CallbackDomain domain_;
  CUpti_CallbackId cbid_;
  int64_t correlationId_;

  // This is only valid during the callback in question
  const CUpti_CallbackData *cbInfo_;

  static std::atomic<id_type> next_id_;
  id_type id_;
  id_type new_id() { return next_id_++; }

  /// Wall time of start and end of API
  cprof::time_point_t wallStart_;
  cprof::time_point_t wallEnd_;

public:
  ApiRecord(const std::string &apiName, const int device,
            const CUpti_CallbackDomain domain, const CUpti_CallbackId cbid,
            const int64_t correlationId, const CUpti_CallbackData *cbInfo)
      : apiName_(apiName), device_(device), domain_(domain), cbid_(cbid),
        correlationId_(correlationId), cbInfo_(cbInfo), id_(new_id()),
        wallStart_(std::chrono::nanoseconds(0)),
        wallEnd_(std::chrono::nanoseconds(0)) {}
  ApiRecord(const int device, const CUpti_CallbackDomain domain,
            const CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
      : ApiRecord(cbInfo->functionName, device, domain, cbid,
                  cbInfo->correlationId, cbInfo) {}
  // Not all ApiRecords come from CUPTI
  ApiRecord(const std::string &apiName, const std::string &kernelName,
            const int device)
      : ApiRecord(apiName, device, CUPTI_CB_DOMAIN_INVALID, -1, -1, nullptr) {
    kernelName_ = kernelName;
  }
  ApiRecord(const std::string &name, const int device)
      : ApiRecord(name, "", device) {}

  void add_input(const cprof::Value &v);
  void add_output(const cprof::Value &v);
  void add_kv(const std::string &key, const std::string &val);
  void add_kv(const std::string &key, const size_t &val);
  void set_wall_start(const cprof::time_point_t &start);
  void set_wall_end(const cprof::time_point_t &end);
  void set_wall_time(const cprof::time_point_t &start,
                     const cprof::time_point_t &end);

  int device() const { return device_; }
  id_type id() const { return id_; }
  const std::string &name() const { return apiName_; }

  std::string json() const;

  bool is_runtime() const { return domain_ == CUPTI_CB_DOMAIN_RUNTIME_API; }
  CUpti_CallbackDomain domain() const { return domain_; }
  CUpti_CallbackId cbid() const { return cbid_; }
  const CUpti_CallbackData *cb_info() const { return cbInfo_; }

  const cprof::time_point_t &wall_end() const { return wallEnd_; }
  const cprof::time_point_t &wall_start() const { return wallStart_; }
};

typedef std::shared_ptr<ApiRecord> ApiRecordRef;

#endif