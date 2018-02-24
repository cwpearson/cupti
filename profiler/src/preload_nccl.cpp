#include <cassert>
#include <dlfcn.h>
#include <mutex>
#include <sstream>

#include <nccl.h>

#include "cprof/allocations.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"
#include "util/logging.hpp"

#include "preload_nccl.hpp"
#include "profiler.hpp"

using cprof::Value;

namespace preload_nccl {
bool passthrough = false;
bool is_passthrough() { return passthrough; }
void set_passthrough(const bool b) { passthrough = b; }
} // namespace preload_nccl

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
    profiler::err() << "ERROR: Unsupported ncclDataType_t" << std::endl;
    assert(0);
  }
}

static void register_ncclBcast(uintptr_t buff, int count,
                               ncclDataType_t datatype, int root,
                               ncclComm_t comm) {
  static std::mutex access;
  static Value rootBuffVal = Value();
  static std::vector<Value> dstBuffVals;
  const int dev = profiler::driver().device(comm);
  const auto &dstAS = profiler::hardware().address_space(dev);
  const size_t numBytes = count * ncclSizeOf(datatype);

  // Only one thread should proceed at a time from here
  std::lock_guard<std::mutex> guard(access);

  auto dstBuffAlloc = profiler::allocations().find(buff, numBytes, dstAS);
  const auto &dstBuffVal = dstBuffAlloc.new_value(buff, numBytes, true);
  assert(dstBuffVal);
  dstBuffVals.push_back(dstBuffVal);

  // If the root has been found, we have enough info to add some deps
  // Have the last thread set the deps and create the api call
  int commSize;
  ncclResult_t res = ncclCommCount(comm, &commSize);
  if (res != ncclSuccess) {
    assert(0);
  }
  if (unsigned(commSize) == dstBuffVals.size()) {

    auto api = std::make_shared<ApiRecord>("ncclBcast", dev);

    // Find the root's buffer value and add it
    const auto &rootDevAS = profiler::hardware().address_space(root);
    rootBuffVal = profiler::allocations().find_value(buff, numBytes, rootDevAS);
    api->add_input(rootBuffVal);

    for (auto &v : dstBuffVals) {
      v.add_depends_on(rootBuffVal, api->id());
      api->add_output(v);
    }
    profiler::atomic_out(api->json());
    dstBuffVals.clear();
    rootBuffVal = Value();
  }
}

static void register_ncclAllReduce(const uintptr_t sendbuff,
                                   const uintptr_t recvbuff, int count,
                                   ncclDataType_t datatype, ncclComm_t comm) {
  static std::mutex access;
  static std::vector<Value> sendBuffVals, recvBuffVals;
  const int dev = profiler::driver().device(comm);
  const auto &AS = profiler::hardware().address_space(dev);

  // Only one thread should proceed at a time from here
  std::lock_guard<std::mutex> guard(access);

  // Look up and add my values
  const size_t numBytes = ncclSizeOf(datatype) * count;
  const auto sendBuffVal =
      profiler::allocations().find_value(sendbuff, numBytes, AS);
  sendBuffVals.push_back(sendBuffVal);

  auto recvBuffAlloc = profiler::allocations().find(recvbuff, numBytes, AS);
  auto recvBuffVal = recvBuffAlloc.new_value(recvbuff, numBytes, true);
  recvBuffVals.push_back(recvBuffVal);

  // Once all values have been found, the last thread to enter allreduce can
  // set up deps
  assert(sendBuffVals.size() == recvBuffVals.size());
  int commSize;
  ncclResult_t res = ncclCommCount(comm, &commSize);
  if (res != ncclSuccess) {
    assert(0);
  }
  if (unsigned(commSize) == sendBuffVals.size()) {

    auto api = std::make_shared<ApiRecord>("ncclAllReduce",
                                           profiler::driver().device(comm));

    for (const auto &sendVal : sendBuffVals) {
      api->add_input(sendVal);
      for (const auto &recvVal : recvBuffVals) {
        recvVal.add_depends_on(sendVal, api->id());
      }
    }
    for (const auto &v : recvBuffVals) {
      api->add_output(v);
    }
    profiler::atomic_out(api->json());
    sendBuffVals.clear();
    recvBuffVals.clear();
  }
}

#define NCCL_DLSYM_BOILERPLATE(name)                                           \
  static name##Func real_##name = nullptr;                                     \
  profiler::err() << "LD_PRELOAD intercept (tid= "                             \
                  << cprof::model::get_thread_id() << "): " << #name           \
                  << std::endl;                                                \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libnccl.so", RTLD_LAZY);                               \
      real_##name = (name##Func)dlsym(h, #name);                               \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");

#define PASSTHROUGH_CHECK(expr)

typedef ncclResult_t (*ncclCommInitAllFunc)(ncclComm_t *comms, int nGPUs,
                                            const int *devList);
extern "C" ncclResult_t ncclCommInitAll(ncclComm_t *comms, int nGPUs,
                                        const int *devList) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitAll);

  if (preload_nccl::is_passthrough()) {
    return real_ncclCommInitAll(comms, nGPUs, devList);
  }

  profiler::err() << "WARN: tid " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during ncclCommInitAll"
                  << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommInitAll(comms, nGPUs, devList);
  for (int i = 0; i < nGPUs; ++i) {
    const int dev = devList ? devList[i] : i;
    profiler::driver().register_ncclComm(comms[i], dev);
  }
  profiler::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef ncclResult_t (*ncclCommInitRankFunc)(ncclComm_t *comm, int ndev,
                                             ncclUniqueId cliqueId, int rank);
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t *comm, int ndev,
                                         ncclUniqueId cliqueId, int rank) {
  NCCL_DLSYM_BOILERPLATE(ncclCommInitRank);

  if (preload_nccl::is_passthrough()) {
    return real_ncclCommInitRank(comm, ndev, cliqueId, rank);
  }

  profiler::err() << "WARN: tid " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during ncclCommInitRank"
                  << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommInitRank(comm, ndev, cliqueId, rank);
  profiler::driver().register_ncclComm(
      *comm, profiler::driver().this_thread().current_device());
  profiler::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef ncclResult_t (*ncclBcastFunc)(void *buff, int count,
                                      ncclDataType_t datatype, int root,
                                      ncclComm_t comm, cudaStream_t stream);
extern "C" ncclResult_t ncclBcast(void *buff, int count,
                                  ncclDataType_t datatype, int root,
                                  ncclComm_t comm, cudaStream_t stream) {
  NCCL_DLSYM_BOILERPLATE(ncclBcast);

  if (preload_nccl::is_passthrough()) {
    return real_ncclBcast(buff, count, datatype, root, comm, stream);
  }

  std::stringstream ss1, ss2, ss3, ss4;
  const auto tid = cprof::model::get_thread_id();
  ss1 << "DEBU: (tid= " << tid << ") bcast register\n";
  ss2 << "DEBU: (tid= " << tid << ") bcast register done\n";
  ss3 << "DEBU: (tid= " << tid << ") bcast call\n";
  ss4 << "DEBU: (tid= " << tid << ") bcast call done\n";

  profiler::err() << "WARN: tid " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during ncclBcast" << std::endl;

  profiler::driver().this_thread().pause_cupti_callbacks();
  logging::atomic_err(ss1.str());
  register_ncclBcast(uintptr_t(buff), count, datatype, root, comm);
  logging::atomic_err(ss2.str());

  logging::atomic_err(ss3.str());
  const ncclResult_t ret =
      real_ncclBcast(buff, count, datatype, root, comm, stream);
  logging::atomic_err(ss4.str());
  profiler::driver().this_thread().resume_cupti_callbacks();
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

  if (preload_nccl::is_passthrough()) {
    return real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                              stream);
  }

  profiler::err() << "WARN: tid " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during ncclAllReduce"
                  << std::endl;

  profiler::driver().this_thread().pause_cupti_callbacks();

  register_ncclAllReduce(uintptr_t(sendbuff), uintptr_t(recvbuff), count,
                         datatype, comm);

  const ncclResult_t ret =
      real_ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
  profiler::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef ncclResult_t (*ncclCommDestroyFunc)(ncclComm_t comm);
extern "C" ncclResult_t nccCommDestroy(ncclComm_t comm) {
  NCCL_DLSYM_BOILERPLATE(ncclCommDestroy);

  profiler::err() << "WARN: tid " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during ncclCommDestroy"
                  << std::endl;

  profiler::driver().this_thread().pause_cupti_callbacks();
  const ncclResult_t ret = real_ncclCommDestroy(comm);
  profiler::driver().this_thread().resume_cupti_callbacks();
  return ret;
}
