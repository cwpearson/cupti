#include <sys/syscall.h>
#include <unistd.h>

#include "cprof/model/thread.hpp"

namespace cprof {
namespace model {

tid_t get_thread_id() { return syscall(SYS_gettid); }
} // namespace model
} // namespace cprof
