#include <sys/syscall.h>
#include <unistd.h>

#include "cprof/model/thread.hpp"

using namespace cprof::model;

tid_t get_thread_id() { return syscall(SYS_gettid); }
