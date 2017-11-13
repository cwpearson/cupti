#include "numa.hpp"

#include "numaif.h"
#include "unistd.h"
#include <cassert>

static void *get_page(const void *ptr, const int page_size) {
  const auto u = reinterpret_cast<uintptr_t>(ptr);
  const auto r = (u / page_size) * page_size;
  return reinterpret_cast<void *>(r);
}

int get_numa_node(const void *ptr) {
  const int page_size = getpagesize();
  void *start_of_page = get_page(ptr, page_size);
  int status = -1;
  long ret_code;
  ret_code =
      move_pages(0 /*self memory */, 1, &start_of_page, NULL, &status, 0);
  assert(!ret_code && "Handle this error.");
  return status;
}

int get_numa_node(const uintptr_t ptr) {
  return get_numa_node(reinterpret_cast<const void *>(ptr));
}