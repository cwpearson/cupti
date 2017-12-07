#include <cassert>
#include <dlfcn.h>

#include <cublas_v2.h>

#include "cprof/allocations.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"

#include "cupti_callbacks.hpp"
#include "profiler.hpp"

using cprof::model::Location;
using cprof::model::Memory;

using profiler::driver;
using profiler::hardware;

#define CUBLAS_DLSYM_BOILERPLATE(name)                                         \
  static name##Func real_##name = nullptr;                                     \
  profiler::err() << "LD_PRELOAD intercept: " #name << std::endl;              \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libcublas.so", RTLD_LAZY);                             \
      real_##name = (name##Func)dlsym(h, #name "_v2");                         \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");

typedef cublasStatus_t (*cublasCreateFunc)(cublasHandle_t *handle);
extern "C" cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  CUBLAS_DLSYM_BOILERPLATE(cublasCreate);

  profiler::err() << "WARN: disabling CUPTI callbacks during cublasCreate call"
                  << std::endl;
  driver().this_thread().pause_cupti_callbacks();

  const cublasStatus_t ret = real_cublasCreate(handle);

  driver().track_cublas_handle(*handle,
                               driver().this_thread().current_device());

  driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cublasStatus_t (*cublasDestroyFunc)(cublasHandle_t handle);
extern "C" cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  CUBLAS_DLSYM_BOILERPLATE(cublasDestroy);

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: tid=" << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during cublasDestroy call"
                  << std::endl;
  const cublasStatus_t ret = real_cublasDestroy(handle);
  driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cublasStatus_t (*cublasDgemmFunc)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc);
extern "C" cublasStatus_t
cublasDgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k, const double *alpha,
            const double *A, int lda, const double *B, int ldb,
            const double *beta, double *C, int ldc) {
  CUBLAS_DLSYM_BOILERPLATE(cublasDgemm);

  // FIXME - also depends on alpha, beta
  // record data, we know things about how this API works

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  auto aVal = allocations.find_value((uintptr_t)A, AS);
  auto bVal = allocations.find_value((uintptr_t)B, AS);
  auto cVal = allocations.find_value((uintptr_t)C, AS);

  assert(aVal && bVal && cVal &&
         "Couldn't find Dgemm argument value on device");

  auto newVal = allocations.duplicate_value(cVal);
  newVal.add_depends_on(aVal);
  newVal.add_depends_on(bVal);
  newVal.add_depends_on(cVal);

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasDgemm call"
                  << std::endl;

  auto api = std::make_shared<ApiRecord>(
      "cublasDgemm", driver().device_from_cublas_handle(handle));
  api->add_output(newVal);
  api->add_input(aVal);
  api->add_input(bVal);
  api->add_input(cVal);
  profiler::atomic_out(api->json());

  const cublasStatus_t ret = real_cublasDgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cublasStatus_t (*cublasSaxpyFunc)(
    cublasHandle_t handle, int n,
    const float *alpha, /* host or device pointer */
    const float *x, int incx, float *y, int incy);
extern "C" cublasStatus_t
cublasSaxpy(cublasHandle_t handle, int n,
            const float *alpha, /* host or device pointer */
            const float *x, int incx, float *y, int incy) {

  CUBLAS_DLSYM_BOILERPLATE(cublasSaxpy);

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find input values
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  auto yVal = allocations.find_value((uintptr_t)y, AS);

  assert(xVal && "Couldn't find cublasSaxpy x value on device");

  // Create output value
  auto outVal = allocations.duplicate_value(yVal);
  outVal.add_depends_on(xVal);
  outVal.add_depends_on(yVal);

  // track api
  auto api = std::make_shared<ApiRecord>(
      "cublasSaxpy", driver().device_from_cublas_handle(handle));
  api->add_output(outVal);
  api->add_input(xVal);
  api->add_input(yVal);
  profiler::atomic_out(api->json());

  // Do the actual call
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasSaxpy call"
                  << std::endl;
  driver().this_thread().pause_cupti_callbacks();
  const cublasStatus_t ret =
      real_cublasSaxpy(handle, n, alpha, x, incx, y, incy);
  driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cublasStatus_t (*cublasSgemmFunc)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, /* host or device pointer */
    const float *A, int lda, const float *B, int ldb,
    const float *beta, /* host or device pointer */
    float *C, int ldc);
extern "C" cublasStatus_t
cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k,
            const float *alpha, /* host or device pointer */
            const float *A, int lda, const float *B, int ldb,
            const float *beta, /* host or device pointer */
            float *C, int ldc) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSgemm);

  // FIXME - also depends on alpha, beta
  // record data, we know things about how this API works

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find the argument values
  auto aVal = allocations.find_value((uintptr_t)A, AS);
  auto bVal = allocations.find_value((uintptr_t)B, AS);
  auto cVal = allocations.find_value((uintptr_t)C, AS);
  assert(aVal && bVal && cVal &&
         "Couldn't find Dgemm argument value on device");

  auto newVal = allocations.duplicate_value(cVal);
  newVal.add_depends_on(aVal);
  newVal.add_depends_on(bVal);
  newVal.add_depends_on(cVal);

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasSgemm "
                     "call"
                  << std::endl;
  const cublasStatus_t ret = real_cublasSgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  driver().this_thread().resume_cupti_callbacks();

  auto api = std::make_shared<ApiRecord>(
      "cublasSgemm", driver().device_from_cublas_handle(handle));
  api->add_output(newVal);
  api->add_input(aVal);
  api->add_input(bVal);
  api->add_input(cVal);
  profiler::atomic_out(api->json());

  return ret;
}

typedef cublasStatus_t (*cublasDgemvFunc)(cublasHandle_t, cublasOperation_t,
                                          int, int, const double *,
                                          const double *, int, const double *,
                                          int, const double *, double *, int);
extern "C" cublasStatus_t cublasDgemv(cublasHandle_t handle,
                                      cublasOperation_t trans, int m, int n,
                                      const double *alpha, const double *A,
                                      int lda, const double *x, int incx,
                                      const double *beta, double *y, int incy) {
  CUBLAS_DLSYM_BOILERPLATE(cublasDgemv);

  // record data, we know things about how this API works

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  auto aVal = allocations.find_value((uintptr_t)A, AS);
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  auto yVal = allocations.find_value((uintptr_t)y, AS);

  assert(aVal && xVal && yVal &&
         "Couldn't find Dgemv argument value on device");

  // FIXME: could use these to do better on dependences
  profiler::err() << "WARN: not handling some values (A, alpha, beta)"
                  << std::endl;

  auto newVal = allocations.duplicate_value(yVal);
  newVal.add_depends_on(xVal);
  newVal.add_depends_on(yVal);

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasDgemv "
                     "call"
                  << std::endl;
  const cublasStatus_t ret = real_cublasDgemv(handle, trans, m, n, alpha, A,
                                              lda, x, incx, beta, y, incy);
  driver().this_thread().resume_cupti_callbacks();

  auto api = std::make_shared<ApiRecord>(
      "cublasDgemv", driver().device_from_cublas_handle(handle));
  api->add_output(newVal);
  api->add_input(aVal);
  api->add_input(xVal);
  api->add_input(yVal);
  profiler::atomic_out(api->json());

  return ret;
}

typedef cublasStatus_t (*cublasSgemvFunc)(cublasHandle_t handle,
                                          cublasOperation_t trans, int m, int n,
                                          const float *alpha, const float *A,
                                          int lda, const float *x, int incx,
                                          const float *beta, float *y,
                                          int incy);
extern "C" cublasStatus_t cublasSgemv(cublasHandle_t handle,
                                      cublasOperation_t trans, int m, int n,
                                      const float *alpha, const float *A,
                                      int lda, const float *x, int incx,
                                      const float *beta, float *y, int incy) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSgemv);

  // record data, we know things about how this API works

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  auto aVal = allocations.find_value((uintptr_t)A, AS);
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  auto yVal = allocations.find_value((uintptr_t)y, AS);

  assert(aVal && xVal && yVal &&
         "Couldn't find cublasSgemv argument value on device");

  // FIXME: could use these to do better on dependences
  profiler::err() << "WARN: not handling some values (A, alpha, beta)"
                  << std::endl;

  auto newVal = allocations.duplicate_value(yVal);
  newVal.add_depends_on(xVal);
  newVal.add_depends_on(yVal);

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasSgemv "
                     "call"
                  << std::endl;
  const cublasStatus_t ret = real_cublasSgemv(handle, trans, m, n, alpha, A,
                                              lda, x, incx, beta, y, incy);
  driver().this_thread().resume_cupti_callbacks();

  auto api = std::make_shared<ApiRecord>(
      "cublasSgemv", driver().device_from_cublas_handle(handle));
  api->add_output(newVal);
  api->add_input(aVal);
  api->add_input(xVal);
  api->add_input(yVal);
  profiler::atomic_out(api->json());

  return ret;
}

typedef cublasStatus_t (*cublasSasumFunc)(cublasHandle_t, int, const float *,
                                          int, float *);
extern "C" cublasStatus_t cublasSasum(cublasHandle_t handle, int n,
                                      const float *x, int incx, float *result) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSasum);

  // record data, we know things about how this API works
  auto &allocations = profiler::allocations();

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  assert(xVal && "Couldn't find Sasum x argument value on device");

  // see if we can find an allocation for the result
  auto rAlloc = allocations.find((uintptr_t)result, sizeof(float), AS);

  if (!rAlloc) {
    // FIXME - can we do a better job with some parameters here
    rAlloc = allocations.new_allocation((uintptr_t)result, sizeof(float), AS,
                                        cprof::model::Memory::Unknown,
                                        Location::Unknown());
    profiler::err() << "WARN: new allocId=" << uintptr_t(rAlloc.id())
                    << " for result=" << uintptr_t(result) << std::endl;
  }
  assert(rAlloc && "If there is no allocation, we need to make one");

  // Make a new value
  auto rVal =
      rAlloc.new_value((uintptr_t)result, sizeof(float), true /*initialized*/);
  rVal.add_depends_on(xVal);

  auto api = std::make_shared<ApiRecord>("cublasSasum", devId);
  api->add_output(rVal);
  api->add_input(xVal);
  profiler::atomic_out(api->json());

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: tid=" << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during cublasSasum call"
                  << std::endl;
  const cublasStatus_t ret = real_cublasSasum(handle, n, x, incx, result);
  driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cublasStatus_t (*cublasSscalFunc)(
    cublasHandle_t handle, int n,
    const float *alpha, /* host or device pointer */
    float *x, int incx);
extern "C" cublasStatus_t
cublasSscal(cublasHandle_t handle, int n,
            const float *alpha, /* host or device pointer */
            float *x, int incx) {
  CUBLAS_DLSYM_BOILERPLATE(cublasSscal);

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find input values
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  assert(xVal && "Couldn't find cublasSscal x value on device");

  // Create output value
  auto outVal = allocations.duplicate_value(xVal);
  outVal.add_depends_on(xVal);

  // track api
  auto api = std::make_shared<ApiRecord>("cublasSscal", devId);
  api->add_output(outVal);
  api->add_input(xVal);
  profiler::atomic_out(api->json());

  // Do the actual call
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasSscal call"
                  << std::endl;
  driver().this_thread().pause_cupti_callbacks();
  const cublasStatus_t ret = real_cublasSscal(handle, n, alpha, x, incx);
  driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cublasStatus_t (*cublasSdotFunc)(cublasHandle_t handle, int n,
                                         const float *x, int incx,
                                         const float *y, int incy,
                                         float *result);
extern "C" cublasStatus_t cublasSdot(cublasHandle_t handle, int n,
                                     const float *x, int incx, const float *y,
                                     int incy, float *result) {

  CUBLAS_DLSYM_BOILERPLATE(cublasSdot);

  // record data, we know things about how this API works
  auto &allocations = profiler::allocations();

  const int devId = driver().device_from_cublas_handle(handle);
  AddressSpace AS = hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  profiler::err() << "Looking for x=" << (uintptr_t)x << std::endl;
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  auto yVal = allocations.find_value((uintptr_t)y, AS);
  assert(xVal && "Couldn't find cublasSdot x argument value on device");
  assert(yVal && "Couldn't find cublasSdot y argument value on device");

  // see if we can find an allocation for the result
  profiler::err() << "Looking for allocation result=" << (uintptr_t)result
                  << std::endl;
  auto rAlloc = allocations.find((uintptr_t)result, sizeof(float), AS);

  if (!rAlloc) {
    profiler::err()
        << "WARN: creating implicit allocation for cublasSdot result"
        << std::endl;
    rAlloc = allocations.new_allocation((uintptr_t)result, sizeof(float), AS,
                                        Memory::Unknown, Location::Unknown());
    assert(rAlloc);
  }
  profiler::err() << "result allocId=" << uintptr_t(rAlloc.id()) << std::endl;
  // Make a new value
  auto rVal =
      rAlloc.new_value((uintptr_t)result, sizeof(float), true /*initialized*/);
  rVal.add_depends_on(xVal);
  rVal.add_depends_on(yVal);

  auto api = std::make_shared<ApiRecord>(
      "cublasSdot", driver().device_from_cublas_handle(handle));
  api->add_output(rVal);
  api->add_input(xVal);
  api->add_input(yVal);
  profiler::atomic_out(api->json());

  driver().this_thread().pause_cupti_callbacks();
  profiler::err() << "WARN: disabling CUPTI callbacks during cublasSdot call"
                  << std::endl;
  const cublasStatus_t ret =
      real_cublasSdot(handle, n, x, incx, y, incy, result);
  driver().this_thread().resume_cupti_callbacks();
  return ret;
}
