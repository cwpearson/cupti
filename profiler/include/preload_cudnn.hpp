#ifndef PRELOAD_CUDNN_HPP
#define PRELOAD_CUDNN_HPP

namespace preload_cudnn {
bool is_passthrough();
void set_passthrough(const bool b);
} // namespace preload_cudnn

#endif