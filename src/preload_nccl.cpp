#include <cassert>
#include <dlfcn.h>
#include <mutex>

#include <nccl.h>

#include "cprof/allocations.hpp"
#include "cprof/apis.hpp"
#include "cprof/callbacks.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"
#include "cprof/profiler.hpp"
#include "cprof/values.hpp"

static size_t ncclSizeOf(const ncclDataType_t t) noexcept {
  switch (t) {
  case ncclChar:
    return sizeof(char);
  case ncclInt:
    return sizeof(int);
  case ncclHalf:
    return 2;
  case ncclFloat:
    return sizeof(float);
  case ncclDouble:
    return sizeof(double);
  case ncclInt64:
    return sizeof(int64_t);
  default:
    assert(0);
  }
}

static void register_ncclBcast(void *buff, int count, ncclDataType_t datatype,
                               int root, ncclComm_t comm) {
  static std::mutex access;
  static Value rootBuffVal = nullptr;
  static std::vector<Value> dstBuffVals;
  const int dev = cprof::driver().device(comm);
  const auto &AS = cprof::hardware().address_space(dev);
  std::lock_guard<std::mutex> guard(access);
  if (dev == root) {
    rootBuffVal = Values::instance().find_live(
        uintptr_t(buff), count * ncclSizeOf(datatype), AS);
  }

  // If the root has been found so far
  if (rootBuffVal) {
    for (const auto &dstBuffVal : dstBuffVals) {
      dstBuffVal->add_depends_on(*rootBuffVal);
    }
    dstBuffVals.clear();
  }
}

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
               << " disabling CUPTI callbacks during ncclCommInitAll"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommInitAll(comms, nGPUs, devList);
  for (int i = 0; i < nGPUs; ++i) {
    const int dev = devList ? devList[i] : i;
    cprof::driver().register_ncclComm(comms[i], dev);
  }
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef ncclResult_t (*ncclCommInitRankFunc)(ncclComm_t *comm, int ndev,
                                             ncclUniqueId cliqueId, int rank);
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t *comm, int ndev,
                                         ncclUniqueId cliqueId, int rank) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitRank);

  cprof::err() << "WARN: tid " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during ncclCommInitRank"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommInitRank(comm, ndev, cliqueId, rank);
  cprof::driver().register_ncclComm(
      *comm, cprof::driver().this_thread().current_device());
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

  register_ncclBcast(buff, count, datatype, root, comm);

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
