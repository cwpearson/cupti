#ifndef PRELOAD_CUBLAS_HPP
#define PRELOAD_CUBLAS_HPP

namespace preload_cublas {
bool is_passthrough();
void set_passthrough(const bool b);
} // namespace preload_cublas

#endif