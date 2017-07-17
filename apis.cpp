
#include "apis.hpp"
#include "output_path.hpp"

const APIs::id_type noid = ApiRecord::noid;

APIs::value_type APIs::insert(const APIs::mapped_type &m) {
  auto id = m->Id();
  auto p = records_.insert(std::make_pair(id, m));

  std::ofstream buf(output_path::get(), std::ofstream::app);
  buf << *m;
  buf.flush();

  return *p.first;
}

APIs::APIs() : records_(map_type()) {}

APIs &APIs::instance() {
  static APIs a;
  return a;
}