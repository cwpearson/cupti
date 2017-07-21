#ifndef APIS_HPP
#define APIS_HPP

#include <fstream>
#include <memory>
#include <mutex>

#include "api_record.hpp"

class APIs {
public:
  typedef ApiRecord::id_type id_type;
  typedef std::shared_ptr<ApiRecord> mapped_type;
  typedef std::pair<id_type, mapped_type> value_type;
  static const id_type noid;

private:
  typedef std::map<id_type, mapped_type> map_type;
  map_type records_;
  std::mutex mutex_;

  value_type _record(const mapped_type &m);

public:
  static APIs &instance();
  static value_type record(const mapped_type &m) {
    return instance()._record(m);
  }

private:
  APIs();
  std::string output_path_;
};

#endif