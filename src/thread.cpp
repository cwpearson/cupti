#include "thread.hpp"

#include <sys/syscall.h>
#include <unistd.h>

tid_t get_thread_id() { return syscall(SYS_gettid); }
