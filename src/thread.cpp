#include <sys/syscall.h>
#include <unistd.h>

#include "cprof/thread.hpp"

tid_t get_thread_id() { return syscall(SYS_gettid); }
