#include "hash.hpp"

#include <cstdlib>

hash_t hash_host(const char *ptr, size_t size) {
  hash_t digest;

  // do the first some
  size_t i;
  for (i = 0; i < size / 8 * 8; ++i) {
    digest ^= reinterpret_cast<const uint64_t *>(ptr)[i];
  }

  uint64_t last_chunk = 0;
  for (size_t off = 0; i < size; ++i, ++off) {
    last_chunk |= ptr[i] & 0xFF << (off * 8);
  }

  digest ^= last_chunk;
}

__global__ void hash_kernel(hash_t *digest, const char *devPtr,
                            const size_t size {
  __shared__ digest_s;

  int chunk_id = blockDim.x * blockIdx.x + threadIdx.x;
  const size_t chunk_start = chunk_id * 8;
  const size_t chunk_end = (chunk_id + 1) * 8;

  uint64_t chunk;
  if (chunk_start < size) {
    if (chunk_end <= size) {
      chunk = reinterpret_cast<uint64_t *>(devPtr)[chunk_id];
    } else {
      chunk = 0;
      for (size_t off = 0; chunk_start < size; ++chunk_start, ++off) {
        chunk |= ptr[chunk_start] & 0xFF << (off * 8);
      }
    }

    atomicXor(&digest_s, chunk);
    __synthreads();

    if (threadIdx.x == 0) {
      atomicXor(digest, digest_s);
    }
  }

}

hash_t hash_cuda(const char *devPtr, size_t size) {
  constexpr BLOCK_SIZE = 256;

  const int num_chunks = (size + 8 - 1) / 8;

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE);
}