#include "hash.hpp"

#include "callbacks.hpp"
#include "check_cuda_error.hpp"

// __global__ void hash_kernel(hash_t *digest, const char *devPtr,
//                             const size_t size) {

//   __shared__ hash_t digest_s;
//   if (threadIdx.x == 0) {
//     digest_s = 0;
//   }
//   __syncthreads();

//   int chunk_id = blockDim.x * blockIdx.x + threadIdx.x;
//   size_t chunk_byte_start = chunk_id * 8;
//   const size_t chunk_byte_end = (chunk_id + 1) * 8;

//   hash_t chunk;
//   if (chunk_byte_start < size) {
//     if (chunk_byte_end <= size) {
//       chunk = reinterpret_cast<const uint64_t *>(devPtr)[chunk_id];
//     } else {
//       chunk = 0;
//       for (size_t off = 0; chunk_byte_start < size; ++chunk_byte_start,
//       ++off) {
//         chunk |= (static_cast<hash_t>(devPtr[chunk_byte_start] & 0xFF))
//                  << (off * 8);
//       }
//     }

//     atomicXor(&digest_s, chunk);
//     __syncthreads();

//     if (threadIdx.x == 0) {
//       atomicXor(digest, digest_s);
//     }
//   }
// }

// hash_t hash_cuda(const char *devPtr, size_t size) {
//   lazyStopCallbacks(); // don't want to profile this
//   constexpr size_t BLOCK_SIZE = 256;

//   const int num_chunks = (size + 8 - 1) / 8;

//   dim3 blockDim(BLOCK_SIZE);
//   dim3 gridDim((num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE);

//   hash_t digest_h = 0;
//   hash_t *digest_d;
//   CUDA_CHECK(cudaMalloc(&digest_d, sizeof(*digest_d)));
//   CUDA_CHECK(cudaMemcpy(digest_d, &digest_h, sizeof(digest_h),
//                         cudaMemcpyHostToDevice));
//   hash_kernel<<<gridDim, blockDim>>>(digest_d, devPtr, size);
//   CUDA_CHECK(cudaGetLastError());
//   CUDA_CHECK(cudaMemcpy(&digest_h, digest_d, sizeof(digest_h),
//                         cudaMemcpyDeviceToHost));

//   lazyActivateCallbacks(); // start profiling stuff again
//   return digest_h;
// }

// hash_t hash_cuda(const uintptr_t ptr, size_t size) {
//   return hash_cuda(reinterpret_cast<const char *>(ptr), size);
// }

hash_t hash_device(const char *devPtr, size_t size) {
  lazyStopCallbacks(); // don't want to profile this

  char *buf = new char[size];
  CUDA_CHECK(cudaMemcpy(buf, devPtr, size, cudaMemcpyDeviceToHost));
  auto digest = hash_host(buf, size);
  delete[] buf;

  lazyActivateCallbacks(); // start profiling stuff again
  return digest;
}

hash_t hash_device(const uintptr_t ptr, size_t size) {
  return hash_device(reinterpret_cast<const char *>(ptr), size);
}

hash_t hash_host(const char *ptr, size_t size) {
  hash_t h = 0;
  for (size_t i = 0; i < size; ++i) {
    h = h * 101 + (hash_t)ptr[i];
  }
  return h;
}

hash_t hash_host(const uintptr_t ptr, size_t size) {
  return hash_host(reinterpret_cast<const char *>(ptr), size);
}