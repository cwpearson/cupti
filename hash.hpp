#ifndef HASH_HPP
#define HASH_HPP

#include <cstdint>

static_assert(sizeof(uint64_t) == sizeof(unsigned long long int),
              "Size no good");

typedef unsigned long long int hash_t;

#endif
