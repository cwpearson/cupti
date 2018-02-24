#ifndef PRELOAD_NCCL_HPP
#define PRELOAD_NCCL_HPP

namespace preload_nccl {
bool is_passthrough();
void set_passthrough(const bool b);
} // namespace preload_nccl

#endif