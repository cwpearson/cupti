#include "cprof/apis.hpp"
#include "cprof/profiler.hpp"

const APIs::id_type noid = ApiRecord::noid;

APIs::value_type APIs::_record(const APIs::mapped_type &m) {
  auto id = m->Id();
  auto p = records_.insert(std::make_pair(id, m));

  logging::atomic_out(m->json());

  return *p.first;
}

APIs::APIs() : records_(map_type()) {}

APIs &APIs::instance() {
  static APIs a;
  return a;
}