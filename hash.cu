#include "hash.hpp"

#include <cstdint>
#include <cstdlib>

__global__ void hash_kernel(hash_t *digest, const char *devPtr,
                            const size_t size) {

  __shared__ hash_t digest_s;

  int chunk_id = blockDim.x * blockIdx.x + threadIdx.x;
  size_t chunk_byte_start = chunk_id * 8;
  const size_t chunk_byte_end = (chunk_id + 1) * 8;

  hash_t chunk;
  if (chunk_byte_start < size) {
    if (chunk_byte_end <= size) {
      chunk = reinterpret_cast<const uint64_t *>(devPtr)[chunk_id];
    } else {
      chunk = 0;
      for (size_t off = 0; chunk_byte_start < size; ++chunk_byte_start, ++off) {
        chunk |= (static_cast<hash_t>(devPtr[chunk_byte_start] & 0xFF))
                 << (off * 8);
      }
    }

    atomicXor(&digest_s, chunk);
    __syncthreads();

    if (threadIdx.x == 0) {
      atomicXor(digest, digest_s);
    }
  }
}

hash_t hash_cuda(const char *devPtr, size_t size) {
  constexpr size_t BLOCK_SIZE = 256;

  const int num_chunks = (size + 8 - 1) / 8;

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim((num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE);

  hash_t digest_h = 0;
  hash_t *digest_d;
  cudaMalloc(&digest_d, sizeof(*digest_d));
  cudaMemcpy(digest_d, &digest_h, sizeof(digest_h), cudaMemcpyHostToDevice);
  hash_kernel<<<gridDim, blockDim>>>(digest_d, devPtr, size);
  cudaMemcpy(&digest_h, digest_d, sizeof(digest_h), cudaMemcpyDeviceToHost);

  return digest_h;
}

hash_t hash_host(const char *ptr, size_t size) {
  hash_t digest = 0;

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
  return digest;
}