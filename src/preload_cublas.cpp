#include <cassert>
#include <dlfcn.h>

#include <cublas_v2.h>

#include "cprof/allocations.hpp"
#include "cprof/apis.hpp"
#include "cprof/callbacks.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"
#include "cprof/preload.hpp"
#include "cprof/values.hpp"

using cprof::model::Location;
using cprof::model::Memory;

typedef cublasStatus_t (*cublasCreateFunc)(cublasHandle_t *handle);
extern "C" cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  V2_LD_PRELOAD_BOILERPLATE(cublasCreate);

  cprof::err() << "WARN: disabling CUPTI callbacks during cublasCreate call"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();

  const cublasStatus_t ret = real_cublasCreate(handle);

  cprof::driver().track_cublas_handle(
      *handle, cprof::driver().this_thread().current_device());

  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cublasStatus_t (*cublasDestroyFunc)(cublasHandle_t handle);
extern "C" cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  V2_LD_PRELOAD_BOILERPLATE(cublasDestroy);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: tid=" << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during cublasDestroy call"
               << std::endl;
  const cublasStatus_t ret = real_cublasDestroy(handle);
  cprof::driver().this_thread().resume_cupti_callbacks();
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
  V2_LD_PRELOAD_BOILERPLATE(cublasDgemm);

  // FIXME - also depends on alpha, beta
  // record data, we know things about how this API works
  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type aId, bId, cId;
  Values::value_type aVal, bVal, cVal;
  std::tie(aId, aVal) = values.find_live((uintptr_t)A, AS);
  std::tie(bId, bVal) = values.find_live((uintptr_t)B, AS);
  std::tie(cId, cVal) = values.find_live((uintptr_t)C, AS);

  assert(aId && bId && cId && "Couldn't find Dgemm argument value on device");

  Values::id_type newId;
  Values::value_type newVal;
  std::tie(newId, newVal) = values.duplicate_value(cVal);
  newVal->add_depends_on(aId);
  newVal->add_depends_on(bId);
  newVal->add_depends_on(cId);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasDgemm call"
               << std::endl;

  auto api = std::make_shared<ApiRecord>(
      "cublasDgemm", cprof::driver().device_from_cublas_handle(handle));
  api->add_output(newId);
  api->add_input(aId);
  api->add_input(bId);
  api->add_input(cId);
  APIs::record(api);

  const cublasStatus_t ret = real_cublasDgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  cprof::driver().this_thread().resume_cupti_callbacks();

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

  V2_LD_PRELOAD_BOILERPLATE(cublasSaxpy);

  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  Values::id_type xId, yId;
  Values::value_type xVal, yVal;
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, AS);
  std::tie(yId, yVal) = values.find_live((uintptr_t)y, AS);

  assert(xId && "Couldn't find cublasSaxpy x value on device");

  // Create output value
  Values::id_type outId;
  Values::value_type outVal;
  std::tie(outId, outVal) = values.duplicate_value(yVal);
  outVal->add_depends_on(xId);
  outVal->add_depends_on(yId);

  // track api
  auto api = std::make_shared<ApiRecord>(
      "cublasSaxpy", cprof::driver().device_from_cublas_handle(handle));
  api->add_output(outId);
  api->add_input(xId);
  api->add_input(yId);
  APIs::record(api);

  // Do the actual call
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasSaxpy call"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cublasStatus_t ret =
      real_cublasSaxpy(handle, n, alpha, x, incx, y, incy);
  cprof::driver().this_thread().resume_cupti_callbacks();

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
  V2_LD_PRELOAD_BOILERPLATE(cublasSgemm);

  // FIXME - also depends on alpha, beta
  // record data, we know things about how this API works
  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find the argument values
  Values::id_type aId, bId, cId;
  Values::value_type aVal, bVal, cVal;
  std::tie(aId, aVal) = values.find_live((uintptr_t)A, AS);
  std::tie(bId, bVal) = values.find_live((uintptr_t)B, AS);
  std::tie(cId, cVal) = values.find_live((uintptr_t)C, AS);

  assert(aId && bId && cId && "Couldn't find Dgemm argument value on device");

  Values::id_type newId;
  Values::value_type newVal;
  std::tie(newId, newVal) = values.duplicate_value(cVal);
  newVal->add_depends_on(aId);
  newVal->add_depends_on(bId);
  newVal->add_depends_on(cId);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasSgemm "
                  "call"
               << std::endl;
  const cublasStatus_t ret = real_cublasSgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  cprof::driver().this_thread().resume_cupti_callbacks();

  auto api = std::make_shared<ApiRecord>(
      "cublasSgemm", cprof::driver().device_from_cublas_handle(handle));
  api->add_output(newId);
  api->add_input(aId);
  api->add_input(bId);
  api->add_input(cId);
  APIs::record(api);

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
  V2_LD_PRELOAD_BOILERPLATE(cublasDgemv);

  // record data, we know things about how this API works
  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type aKey, xKey, yKey;
  Values::value_type aVal, xVal, yVal;
  std::tie(aKey, aVal) = values.find_live((uintptr_t)A, AS);
  std::tie(xKey, xVal) = values.find_live((uintptr_t)x, AS);
  std::tie(yKey, yVal) = values.find_live((uintptr_t)y, AS);

  assert(aKey && xKey && yKey &&
         "Couldn't find Dgemv argument value on device");

  // FIXME: could use these to do better on dependences
  cprof::err() << "WARN: not handling some values (A, alpha, beta)"
               << std::endl;

  Values::id_type newId;
  Values::value_type newVal;
  std::tie(newId, newVal) = values.duplicate_value(yVal);
  newVal->add_depends_on(xKey);
  newVal->add_depends_on(yKey);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasDgemv "
                  "call"
               << std::endl;
  const cublasStatus_t ret = real_cublasDgemv(handle, trans, m, n, alpha, A,
                                              lda, x, incx, beta, y, incy);
  cprof::driver().this_thread().resume_cupti_callbacks();

  auto api = std::make_shared<ApiRecord>(
      "cublasDgemv", cprof::driver().device_from_cublas_handle(handle));
  api->add_output(newId);
  api->add_input(aKey);
  api->add_input(xKey);
  api->add_input(yKey);
  APIs::record(api);

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
  V2_LD_PRELOAD_BOILERPLATE(cublasSgemv);

  // record data, we know things about how this API works
  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type aKey, xKey, yKey;
  Values::value_type aVal, xVal, yVal;
  std::tie(aKey, aVal) = values.find_live((uintptr_t)A, AS);
  std::tie(xKey, xVal) = values.find_live((uintptr_t)x, AS);
  std::tie(yKey, yVal) = values.find_live((uintptr_t)y, AS);

  assert(aKey && xKey && yKey &&
         "Couldn't find cublasSgemv argument value on device");

  // FIXME: could use these to do better on dependences
  cprof::err() << "WARN: not handling some values (A, alpha, beta)"
               << std::endl;

  Values::id_type newId;
  Values::value_type newVal;
  std::tie(newId, newVal) = values.duplicate_value(yVal);
  newVal->add_depends_on(xKey);
  newVal->add_depends_on(yKey);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasSgemv "
                  "call"
               << std::endl;
  const cublasStatus_t ret = real_cublasSgemv(handle, trans, m, n, alpha, A,
                                              lda, x, incx, beta, y, incy);
  cprof::driver().this_thread().resume_cupti_callbacks();

  auto api = std::make_shared<ApiRecord>(
      "cublasSgemv", cprof::driver().device_from_cublas_handle(handle));
  api->add_output(newId);
  api->add_input(aKey);
  api->add_input(xKey);
  api->add_input(yKey);
  APIs::record(api);

  return ret;
}

typedef cublasStatus_t (*cublasSasumFunc)(cublasHandle_t, int, const float *,
                                          int, float *);
extern "C" cublasStatus_t cublasSasum(cublasHandle_t handle, int n,
                                      const float *x, int incx, float *result) {
  V2_LD_PRELOAD_BOILERPLATE(cublasSasum);

  // record data, we know things about how this API works
  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type xId;
  std::tie(xId, std::ignore) = values.find_live((uintptr_t)x, AS);
  assert(xId && "Couldn't find Sasum x argument value on device");

  // see if we can find an allocation for the result
  auto rAlloc = allocations.find((uintptr_t)result, sizeof(float), AS);

  if (!rAlloc) {
    // FIXME - can we do a better job with some parameters here
    rAlloc = allocations.new_allocation((uintptr_t)result, sizeof(float), AS,
                                        cprof::model::Memory::Unknown,
                                        Location::Unknown());
    cprof::err() << "WARN: new allocId=" << uintptr_t(rAlloc.get())
                 << " for result=" << uintptr_t(result) << std::endl;
  }
  assert(rAlloc && "If there is no allocation, we need to make one");

  // Make a new value
  Values::id_type rId;
  Values::value_type rVal;
  std::tie(rId, rVal) =
      values.new_value((uintptr_t)result, sizeof(float), rAlloc);
  rVal->add_depends_on(xId);

  auto api = std::make_shared<ApiRecord>("cublasSasum", devId);
  api->add_output(rId);
  api->add_input(xId);
  APIs::record(api);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: tid=" << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during cublasSasum call"
               << std::endl;
  const cublasStatus_t ret = real_cublasSasum(handle, n, x, incx, result);
  cprof::driver().this_thread().resume_cupti_callbacks();
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
  V2_LD_PRELOAD_BOILERPLATE(cublasSscal);

  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  Values::id_type xId, outId;
  Values::value_type xVal, outVal;
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, AS);
  assert(xId && "Couldn't find cublasSscal x value on device");

  // Create output value
  std::tie(outId, outVal) = values.duplicate_value(xVal);
  outVal->add_depends_on(xId);

  // track api
  auto api = std::make_shared<ApiRecord>("cublasSscal", devId);
  api->add_output(outId);
  api->add_input(xId);
  APIs::record(api);

  // Do the actual call
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasSscal call"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cublasStatus_t ret = real_cublasSscal(handle, n, alpha, x, incx);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cublasStatus_t (*cublasSdotFunc)(cublasHandle_t handle, int n,
                                         const float *x, int incx,
                                         const float *y, int incy,
                                         float *result);
extern "C" cublasStatus_t cublasSdot(cublasHandle_t handle, int n,
                                     const float *x, int incx, const float *y,
                                     int incy, float *result) {

  V2_LD_PRELOAD_BOILERPLATE(cublasSdot);

  // record data, we know things about how this API works
  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  const int devId = cprof::driver().device_from_cublas_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find the argument values
  // http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
  Values::id_type xId, yId;
  cprof::err() << "Looking for x=" << (uintptr_t)x << std::endl;
  std::tie(xId, std::ignore) = values.find_live((uintptr_t)x, AS);
  assert(xId && "Couldn't find cublasSdot x argument value on device");
  std::tie(yId, std::ignore) = values.find_live((uintptr_t)y, AS);
  assert(yId && "Couldn't find cublasSdot y argument value on device");

  // see if we can find an allocation for the result
  cprof::err() << "Looking for allocation result=" << (uintptr_t)result
               << std::endl;
  auto rAlloc = allocations.find((uintptr_t)result, sizeof(float), AS);

  if (!rAlloc) {
    cprof::err() << "WARN: creating implicit allocation for cublasSdot result"
                 << std::endl;
    rAlloc = allocations.new_allocation((uintptr_t)result, sizeof(float), AS,
                                        Memory::Unknown, Location::Unknown());
    assert(rAlloc);
  }
  cprof::err() << "result allocId=" << uintptr_t(rAlloc.get()) << std::endl;
  // Make a new value
  Values::id_type rId;
  Values::value_type rVal;
  std::tie(rId, rVal) =
      values.new_value((uintptr_t)result, sizeof(float), rAlloc);
  rVal->add_depends_on(xId);
  rVal->add_depends_on(yId);

  auto api = std::make_shared<ApiRecord>(
      "cublasSdot", cprof::driver().device_from_cublas_handle(handle));
  api->add_output(rId);
  api->add_input(xId);
  api->add_input(yId);
  APIs::record(api);

  cprof::driver().this_thread().pause_cupti_callbacks();
  cprof::err() << "WARN: disabling CUPTI callbacks during cublasSdot call"
               << std::endl;
  const cublasStatus_t ret =
      real_cublasSdot(handle, n, x, incx, y, incy, result);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}
