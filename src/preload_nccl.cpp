#include <cassert>
#include <dlfcn.h>

#include <nccl.h>

#include "cprof/allocations.hpp"
#include "cprof/apis.hpp"
#include "cprof/callbacks.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"
#include "cprof/profiler.hpp"
#include "cprof/values.hpp"

#define NCCL_DLSYM_BOILERPLATE(name)                                           \
  static name##Func real_##name = nullptr;                                     \
  cprof::err() << "LD_PRELOAD intercept: " #name << std::endl;                 \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libnccl.so", RTLD_LAZY);                               \
      real_##name = (name##Func)dlsym(h, #name);                               \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");

typedef ncclResult_t (*ncclCommInitAllFunc)(ncclComm_t *comms, int nGPUs,
                                            const int *devList);
extern "C" ncclResult_t ncclCommInitAll(ncclComm_t *comms, int nGPUs,
                                        const int *devList) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitAll);
  cprof::err() << "WARN: tid " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during ncclBCommInitAll"
               << std::endl;

  cprof::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommInitAll(comms, nGPUs, devList);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef ncclResult_t (*ncclCommInitRankFunc)(ncclComm_t *comm, int nGPUs,
                                             ncclUniqueId cliqueId, int rank);
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nGPUs,
                                         ncclUniqueId cliqueId, int rank) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitRank);
  cprof::err() << "WARN: tid " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during ncclCommInitRank"
               << std::endl;

  cprof::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommInitRank(comm, nGPUs, cliqueId, rank);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef ncclResult_t (*ncclBcastFunc)(void *buff, int count,
                                      ncclDataType_t datatype, int root,
                                      ncclComm_t comm, cudaStream_t stream);
extern "C" ncclResult_t ncclBcast(void *buff, int count,
                                  ncclDataType_t datatype, int root,
                                  ncclComm_t comm, cudaStream_t stream) {
  NCCL_DLSYM_BOILERPLATE(ncclBcast);

  cprof::err() << "WARN: tid " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during ncclBcast" << std::endl;

  // current_device buf deps on root buff
  // const int devId = cprof::driver().current_device();

  // auto rootBufVal = Values::instance().find();

  cprof::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret =
      real_ncclBcast(buff, count, datatype, root, comm, stream);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef ncclResult_t (*ncclAllReduceFunc)(const void *sendbuff, void *recvbuff,
                                          int count, ncclDataType_t datatype,
                                          ncclRedOp_t op, ncclComm_t comm,
                                          cudaStream_t stream);

extern "C" ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff,
                                      int count, ncclDataType_t datatype,
                                      ncclRedOp_t op, ncclComm_t comm,
                                      cudaStream_t stream) {
  NCCL_DLSYM_BOILERPLATE(ncclAllReduce);

  cprof::err() << "WARN: tid " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during ncclAllReduce"
               << std::endl;

  cprof::err() << "WARN: not doing anything with ncclAllReduce" << std::endl;

  cprof::driver().this_thread().pause_cupti_callbacks();

  const ncclResult_t ret =
      real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}
