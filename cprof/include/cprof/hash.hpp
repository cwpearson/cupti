#ifndef HASH_HPP
#define HASH_HPP

#include <cstdint>
#include <cstdlib>

static_assert(sizeof(uint64_t) == sizeof(unsigned long long int),
              "Size no good");

typedef unsigned long long int hash_t;

// hash_t hash_cuda(const char *devPtr, size_t size);
// hash_t hash_cuda(const uintptr_t devPtr, size_t size);

hash_t hash_host(const char *ptr, size_t size);
hash_t hash_host(const uintptr_t ptr, size_t size);

hash_t hash_device(const char *devPtr, size_t size);
hash_t hash_device(const uintptr_t devPtr, size_t size);

#endif
