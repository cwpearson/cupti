
#include <cassert>
#include <dlfcn.h>
#include <list>

#include <cudnn.h>

#include "cprof/allocations.hpp"
#include "cprof/apis.hpp"
#include "cprof/callbacks.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"
#include "cprof/preload.hpp"
#include "cprof/values.hpp"

typedef cudnnStatus_t (*cudnnCreateFunc)(cudnnHandle_t *handle);
extern "C" cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
  SAME_LD_PRELOAD_BOILERPLATE(cudnnCreate);

  cprof::err() << "WARN: tid " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during cudnnCreate call"
               << std::endl;

  cprof::driver().this_thread().pause_cupti_callbacks();

  const cudnnStatus_t ret = real_cudnnCreate(handle);
  cprof::driver().track_cudnn_handle(
      *handle, cprof::driver().this_thread().current_device());
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cudnnStatus_t (*cudnnDestroyFunc)(cudnnHandle_t handle);
extern "C" cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  SAME_LD_PRELOAD_BOILERPLATE(cudnnDestroy);

  cprof::err() << "WARN: disabling CUPTI callbacks during cudnnDestroy call"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();

  const cudnnStatus_t ret = real_cudnnDestroy(handle);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cudnnStatus_t (*cudnnActivationForwardFunc)(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);

extern "C" cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {

  // FIXME - also depends on alpha, beta
  SAME_LD_PRELOAD_BOILERPLATE(cudnnActivationForward);

  Values::id_type xId, yId;
  Values::value_type xVal, yVal;

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Get src value
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, 1, AS);
  assert(xId && "x should be on device");

  // Get dst allocation
  auto yAlloc = allocations.find((uintptr_t)y, AS);
  assert(yAlloc && "y alloc should be on device");

  std::tie(yId, yVal) = values.new_value((uintptr_t)y, 0, yAlloc, true);
  yVal->add_depends_on(xId);

  auto api = std::make_shared<ApiRecord>("cudnnActivationForward", devId);
  api->add_output(yId);
  api->add_input(xId);
  APIs::record(api);

  cprof::err()
      << "WARN: disabling CUPTI callbacks during cudnnActivationForward call"
      << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnActivationForward(
      handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cudnnStatus_t (*cudnnAddTensorFunc)(cudnnHandle_t handle,
                                            const void *alpha,
                                            const cudnnTensorDescriptor_t aDesc,
                                            const void *A, const void *beta,
                                            const cudnnTensorDescriptor_t cDesc,
                                            void *C);
extern "C" cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle, const void *alpha,
                                        const cudnnTensorDescriptor_t aDesc,
                                        const void *A, const void *beta,
                                        const cudnnTensorDescriptor_t cDesc,
                                        void *C) {
  SAME_LD_PRELOAD_BOILERPLATE(cudnnAddTensor);

  // FIXME - alpha and beta

  Values::id_type aId, cId;
  Values::value_type aVal, cVal;

  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Get src value
  std::tie(aId, aVal) = values.find_live((uintptr_t)A, 1, AS);
  assert(aId && "A should be on device");
  std::tie(cId, cVal) = values.find_live((uintptr_t)C, 1, AS);
  assert(cId && "C should be on device");

  Values::id_type dstId;
  Values::value_type dstVal;
  std::tie(dstId, dstVal) = values.duplicate_value(cVal);
  dstVal->add_depends_on(aId);
  dstVal->add_depends_on(cId);

  auto api = std::make_shared<ApiRecord>("cudnnAddTensor", devId);
  api->add_output(dstId);
  api->add_input(aId);
  api->add_input(cId);
  APIs::record(api);

  cprof::err() << "WARN: thread " << cprof::model::get_thread_id()
               << " disabling CUPTI callbacks during cudnnAddTensor call"
               << std::endl;

  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  cprof::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cudnnStatus_t (*cudnnActivationBackwardFunc)(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx);
extern "C" cudnnStatus_t cudnnActivationBackward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t yDesc, const void *y,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnTensorDescriptor_t xDesc, const void *x, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {

  SAME_LD_PRELOAD_BOILERPLATE(cudnnActivationBackward);

  Values::id_type yId, dyId, xId, dxId;
  Values::value_type yVal, dyVal, xVal, dxVal;

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Get src value
  std::tie(yId, yVal) = values.find_live((uintptr_t)y, 1, AS);
  assert(yId && "y should be on device");
  std::tie(dyId, dyVal) = values.find_live((uintptr_t)dy, 1, AS);
  assert(dyId && "dy should be on device");
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, 1, AS);
  assert(xId && "x should be on device");

  // Get dst allocation
  auto dxAlloc = allocations.find((uintptr_t)dx, AS);
  assert(dxAlloc && "dx alloc should be on device");

  // FIXME - this size is wrong
  std::tie(dxId, dxVal) = values.new_value((uintptr_t)dx, 0, dxAlloc, true);
  dxVal->add_depends_on(xId);
  dxVal->add_depends_on(yId);
  dxVal->add_depends_on(dyId);

  // FIXME: also depends on alpha, beta
  auto api = std::make_shared<ApiRecord>(
      "cudnnActivationBackward",
      cprof::driver().device_from_cudnn_handle(handle));
  api->add_output(dxId);
  api->add_input(xId);
  api->add_input(yId);
  api->add_input(dyId);
  APIs::record(api);

  cprof::err()
      << "WARN: disabling CUPTI callbacks during cudnnActivationBackward call"
      << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y,
                                   dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionBackwardDataFunc)(
    cudnnHandle_t, const void *, const cudnnFilterDescriptor_t, const void *,
    const cudnnTensorDescriptor_t, const void *,
    const cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, void *,
    size_t, const void *, const cudnnTensorDescriptor_t, void *);
extern "C" cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void *alpha,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t dxDesc, void *dx) {
  SAME_LD_PRELOAD_BOILERPLATE(cudnnConvolutionBackwardData);
  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  Values::id_type wId, dyId, workSpaceId, dxId;
  Values::value_type wVal, dyVal, workSpaceVal, dxVal;
  std::tie(dyId, dyVal) = values.find_live((uintptr_t)dy, AS);
  std::tie(wId, wVal) = values.find_live((uintptr_t)w, AS);
  std::tie(workSpaceId, workSpaceVal) =
      values.find_live((uintptr_t)workSpace, AS);
  std::tie(dxId, dxVal) = values.find_live((uintptr_t)dx, AS);

  assert(dyId &&
         "Couldn't find cudnnConvolutionBackwardData dy value on device");

  // Create output value
  Values::id_type outId;
  Values::value_type outVal;
  std::tie(outId, outVal) = values.duplicate_value(dxVal);
  outVal->add_depends_on(wId);
  outVal->add_depends_on(dyId);
  outVal->add_depends_on(workSpaceId);
  outVal->add_depends_on(dxId);
  // track api
  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionBackwardData",
      cprof::driver().device_from_cudnn_handle(handle));
  api->add_output(outId);
  api->add_input(wId);
  api->add_input(dyId);
  api->add_input(workSpaceId);
  api->add_input(dxId);
  APIs::record(api);

  // Do the actual call
  cprof::err() << "WARN: disabling CUPTI callbacks during "
                  "cudnnConvolutionBackwardData call"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardData(
      handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dxDesc, dx);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionBackwardBiasFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t dyDesc, const void *dy, const void *beta,
    const cudnnTensorDescriptor_t dbDesc, void *db);
extern "C" cudnnStatus_t
cudnnConvolutionBackwardBias(cudnnHandle_t handle, const void *alpha,
                             const cudnnTensorDescriptor_t dyDesc,
                             const void *dy, const void *beta,
                             const cudnnTensorDescriptor_t dbDesc, void *db) {
  SAME_LD_PRELOAD_BOILERPLATE(cudnnConvolutionBackwardBias);
  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  Values::id_type dyId, dbId;
  Values::value_type dyVal, dbVal;
  std::tie(dyId, dyVal) = values.find_live((uintptr_t)dy, AS);

  assert(dyId &&
         "Couldn't find cudnnConvolutionBackwardBias dy value on device");

  // Create output value
  auto dbAlloc = allocations.find((uintptr_t)db, 1, AS);
  assert(dbAlloc && "y allocation should be on device");
  std::tie(dbId, dbVal) = values.new_value((uintptr_t)db, 0, dbAlloc);
  dbVal->add_depends_on(dyId);

  // track api
  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionBackwardBias",
      cprof::driver().device_from_cudnn_handle(handle));
  api->add_output(dbId);
  api->add_input(dyId);
  APIs::record(api);

  // Do the actual call
  cprof::err() << "WARN: disabling CUPTI callbacks during "
                  "cudnnConvolutionBackwardBias call"
               << std::endl;

  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardBias(
      handle, alpha, dyDesc, dy, beta, dbDesc, db);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionBackwardFilterFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw);
extern "C" cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnTensorDescriptor_t dyDesc, const void *dy,
    const cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void *workSpace,
    size_t workSpaceSizeInBytes, const void *beta,
    const cudnnFilterDescriptor_t dwDesc, void *dw) {

  SAME_LD_PRELOAD_BOILERPLATE(cudnnConvolutionBackwardFilter);
  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  Values::id_type xId, dyId, workSpaceId, dwId;
  Values::value_type dwVal;
  std::tie(xId, std::ignore) = values.find_live((uintptr_t)x, AS);
  std::tie(dyId, std::ignore) = values.find_live((uintptr_t)dy, AS);
  std::tie(workSpaceId, std::ignore) =
      values.find_live((uintptr_t)workSpace, AS);
  std::tie(dwId, dwVal) = values.find_live((uintptr_t)dw, AS);
  assert(
      xId && dyId && workSpaceId && dwId &&
      "Couldn't find cudnnConvolutionBackwardFilter argument value on device");

  // See if there is an existing output value to take info from
  Values::id_type outId;
  Values::value_type outVal;
  std::tie(outId, outVal) = values.duplicate_value(dwVal);
  outVal->add_depends_on(xId);
  outVal->add_depends_on(dyId);
  outVal->add_depends_on(workSpaceId);
  outVal->add_depends_on(dwId);
  cprof::err() << "[cudnnConvolutionBackwardFilter] " << outId << " deps on "
               << xId << " " << dyId << " " << workSpaceId << " " << dwId
               << std::endl;

  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionForward",
      cprof::driver().device_from_cudnn_handle(handle));
  api->add_output(outId);
  api->add_input(xId);
  api->add_input(dyId);
  api->add_input(workSpaceId);
  api->add_input(dwId);
  APIs::record(api);

  cprof::err() << "WARN: disabling CUPTI callbacks during "
                  "cudnnConvolutionBackwardFilter call"
               << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardFilter(
      handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dwDesc, dw);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cudnnStatus_t (*cudnnConvolutionForwardFunc)(
    cudnnHandle_t handle, const void *alpha,
    const cudnnTensorDescriptor_t xDesc, const void *x,
    const cudnnFilterDescriptor_t wDesc, const void *w,
    const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void *workSpace, size_t workSpaceSizeInBytes, const void *beta,
    const cudnnTensorDescriptor_t yDesc, void *y);

extern "C" cudnnStatus_t
cudnnConvolutionForward(cudnnHandle_t handle, const void *alpha,
                        const cudnnTensorDescriptor_t xDesc, const void *x,
                        const cudnnFilterDescriptor_t wDesc, const void *w,
                        const cudnnConvolutionDescriptor_t convDesc,
                        cudnnConvolutionFwdAlgo_t algo, void *workSpace,
                        size_t workSpaceSizeInBytes, const void *beta,
                        const cudnnTensorDescriptor_t yDesc, void *y) {

  SAME_LD_PRELOAD_BOILERPLATE(cudnnConvolutionForward);

  auto &values = Values::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  cprof::err() << "Looking for x=" << (uintptr_t)x << ", w=" << (uintptr_t)w
               << ", workSpace=" << (uintptr_t)workSpace << std::endl;
  Values::id_type xId, wId, workSpaceId, yId;
  Values::value_type yVal;
  std::tie(xId, std::ignore) = values.find_live((uintptr_t)x, AS);
  std::tie(wId, std::ignore) = values.find_live((uintptr_t)w, AS);
  std::tie(workSpaceId, std::ignore) =
      values.find_live((uintptr_t)workSpace, AS);
  std::tie(yId, yVal) = values.find_live((uintptr_t)y, AS);
  assert(xId && wId && workSpaceId && yId &&
         "Couldn't find cudnnConvolutionForward argument value on device");

  // See if there is an existing output value to take info from
  Values::id_type outId;
  Values::value_type outVal;
  std::tie(outId, outVal) = values.duplicate_value(yVal);
  outVal->add_depends_on(xId);
  outVal->add_depends_on(wId);
  outVal->add_depends_on(workSpaceId);
  outVal->add_depends_on(yId);
  cprof::err() << "[cudnnConvolutionForward] " << outId << " deps on " << yId
               << " " << xId << " " << wId << " " << workSpaceId << std::endl;

  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionForward",
      cprof::driver().device_from_cudnn_handle(handle));
  api->add_output(outId);
  api->add_input(xId);
  api->add_input(wId);
  api->add_input(workSpaceId);
  api->add_input(yId);
  APIs::record(api);

  cprof::err()
      << "WARN: thread " << cprof::model::get_thread_id()
      << " disabling CUPTI callbacks during cudnnConvolutionForward call"
      << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionForward(
      handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, yDesc, y);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

typedef cudnnStatus_t (*cudnnSoftmaxForwardFunc)(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
extern "C" cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algo, cudnnSoftmaxMode_t mode,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {

  SAME_LD_PRELOAD_BOILERPLATE(cudnnSoftmaxForward);

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  const int devId = cprof::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = cprof::hardware().address_space(devId);

  // Find input values
  Values::id_type xId, yId;
  Values::value_type xVal, yVal;
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, AS);

  assert(xId && "Couldn't find cudnnSoftmaxForward x value on device");

  // Create output value
  auto yAlloc = allocations.find((uintptr_t)y, 1, AS);
  assert(yAlloc && "y allocation should be on device");
  std::tie(yId, yVal) = values.new_value((uintptr_t)y, 0, yAlloc);
  yVal->add_depends_on(xId);

  // track api
  auto api = std::make_shared<ApiRecord>("cudnnSoftmaxForward", devId);
  api->add_output(yId);
  api->add_input(xId);
  APIs::record(api);

  // Do the actual call
  cprof::err()
      << "WARN: disabling CUPTI callbacks during cudnnSoftmaxForward call"
      << std::endl;
  cprof::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnSoftmaxForward(handle, algo, mode, alpha,
                                                     xDesc, x, beta, yDesc, y);
  cprof::driver().this_thread().resume_cupti_callbacks();

  return ret;
}
