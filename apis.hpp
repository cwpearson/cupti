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

public:
  value_type insert(const mapped_type &m);

  static APIs &instance();

private:
  APIs();
  std::string output_path_;
};

#endif