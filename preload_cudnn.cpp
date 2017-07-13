
#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <list>

#include <cudnn.h>

#include "allocations.hpp"
#include "apis.hpp"
#include "callbacks.hpp"
#include "driver_state.hpp"
#include "preload.hpp"
#include "thread.hpp"
#include "values.hpp"

typedef cudnnStatus_t (*cudnnActivationForwardFunc)(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);

extern "C" cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {

  // FIXME - also depends on alpha, beta
  CUDNN_LD_PRELOAD_BOILERPLATE(cudnnActivationForward);

  Values::id_type xId, yId;
  Values::value_type xVal, yVal;

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Get src value
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, 1, AddressSpace::Cuda());
  assert(xId && "x should be on device");

  // Get dst allocation
  Allocations::id_type yAllocId;
  std::tie(yAllocId, std::ignore) =
      allocations.find_live((uintptr_t)y, AddressSpace::Cuda());
  assert(yAllocId && "y alloc should be on device");

  std::tie(yId, yVal) = values.new_value((uintptr_t)y, 0, yAllocId, true);
  yVal->add_depends_on(xId);

  auto api = std::make_shared<ApiRecord>(
      "cudnnActivationForward", DriverState::this_thread().current_device());
  api->add_output(yId);
  api->add_input(xId);
  APIs::instance().insert(api);

  printf(
      "WARN: disabling CUPTI callbacks during cudnnActivationForward call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnActivationForward(
      handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  DriverState::this_thread().resume_cupti_callbacks();

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
  CUDNN_LD_PRELOAD_BOILERPLATE(cudnnAddTensor);

  // FIXME - alpha and beta

  Values::id_type aId, cId;
  Values::value_type aVal, cVal;

  auto &values = Values::instance();

  // Get src value
  std::tie(aId, aVal) = values.find_live((uintptr_t)A, 1, AddressSpace::Cuda());
  assert(aId && "A should be on device");
  std::tie(cId, cVal) = values.find_live((uintptr_t)C, 1, AddressSpace::Cuda());
  assert(cId && "C should be on device");

  Values::id_type dstId;
  Values::value_type dstVal;
  std::tie(dstId, dstVal) = values.duplicate_value(cVal);
  dstVal->add_depends_on(aId);
  dstVal->add_depends_on(cId);

  auto api = std::make_shared<ApiRecord>(
      "cudnnAddTensor", DriverState::this_thread().current_device());
  api->add_output(dstId);
  api->add_input(aId);
  api->add_input(cId);
  APIs::instance().insert(api);

  printf("WARN: disabling CUPTI callbacks during cudnnAddTensor call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  DriverState::this_thread().resume_cupti_callbacks();
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

  CUDNN_LD_PRELOAD_BOILERPLATE(cudnnActivationBackward);

  Values::id_type yId, dyId, xId, dxId;
  Values::value_type yVal, dyVal, xVal, dxVal;

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Get src value
  std::tie(yId, yVal) = values.find_live((uintptr_t)y, 1, AddressSpace::Cuda());
  assert(yId && "y should be on device");
  std::tie(dyId, dyVal) =
      values.find_live((uintptr_t)dy, 1, AddressSpace::Cuda());
  assert(dyId && "dy should be on device");
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, 1, AddressSpace::Cuda());
  assert(xId && "x should be on device");

  // Get dst allocation
  Allocations::id_type dxAllocId;
  std::tie(dxAllocId, std::ignore) =
      allocations.find_live((uintptr_t)dx, AddressSpace::Cuda());
  assert(dxAllocId && "dx alloc should be on device");

  // FIXME - this size is wrong
  std::tie(dxId, dxVal) = values.new_value((uintptr_t)dx, 0, dxAllocId, true);
  dxVal->add_depends_on(xId);
  dxVal->add_depends_on(yId);
  dxVal->add_depends_on(dyId);

  // FIXME: also depends on alpha, beta
  auto api = std::make_shared<ApiRecord>(
      "cudnnActivationBackward", DriverState::this_thread().current_device());
  api->add_output(dxId);
  api->add_input(xId);
  api->add_input(yId);
  api->add_input(dyId);
  APIs::instance().insert(api);

  printf(
      "WARN: disabling CUPTI callbacks during cudnnActivationBackward call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y,
                                   dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  DriverState::this_thread().resume_cupti_callbacks();

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

  CUDNN_LD_PRELOAD_BOILERPLATE(cudnnConvolutionBackwardFilter);
  auto &values = Values::instance();

  // Find input values

  Values::id_type xId, dyId, workSpaceId, dwId;
  Values::value_type dwVal;
  std::tie(xId, std::ignore) =
      values.find_live((uintptr_t)x, AddressSpace::Cuda());
  std::tie(dyId, std::ignore) =
      values.find_live((uintptr_t)dy, AddressSpace::Cuda());
  std::tie(workSpaceId, std::ignore) =
      values.find_live((uintptr_t)workSpace, AddressSpace::Cuda());
  std::tie(dwId, dwVal) = values.find_live((uintptr_t)dw, AddressSpace::Cuda());
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
  printf("[cudnnConvolutionBackwardFilter] %lu deps on %lu %lu %lu %lu\n",
         outId, xId, dyId, workSpaceId, dwId);

  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionForward", DriverState::this_thread().current_device());
  api->add_output(outId);
  api->add_input(xId);
  api->add_input(dyId);
  api->add_input(workSpaceId);
  api->add_input(dwId);
  APIs::instance().insert(api);

  printf("WARN: disabling CUPTI callbacks during "
         "cudnnConvolutionBackwardFilter call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardFilter(
      handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dwDesc, dw);
  DriverState::this_thread().resume_cupti_callbacks();

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

  CUDNN_LD_PRELOAD_BOILERPLATE(cudnnConvolutionForward);

  auto &values = Values::instance();

  // Find input values
  printf("Looking for x=%lu, w=%lu, workSpace=%lu\n", (uintptr_t)x,
         (uintptr_t)w, (uintptr_t)workSpace);
  Values::id_type xId, wId, workSpaceId, yId;
  Values::value_type yVal;
  std::tie(xId, std::ignore) =
      values.find_live((uintptr_t)x, AddressSpace::Cuda());
  std::tie(wId, std::ignore) =
      values.find_live((uintptr_t)w, AddressSpace::Cuda());
  std::tie(workSpaceId, std::ignore) =
      values.find_live((uintptr_t)workSpace, AddressSpace::Cuda());
  std::tie(yId, yVal) = values.find_live((uintptr_t)y, AddressSpace::Cuda());
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
  printf("[cudnnConvolutionForward] %lu deps on %lu %lu %lu %lu\n", outId, yId,
         xId, wId, workSpaceId);

  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionForward", DriverState::this_thread().current_device());
  api->add_output(outId);
  api->add_input(xId);
  api->add_input(wId);
  api->add_input(workSpaceId);
  api->add_input(yId);
  APIs::instance().insert(api);

  printf(
      "WARN: disabling CUPTI callbacks during cudnnConvolutionForward call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionForward(
      handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, yDesc, y);
  DriverState::this_thread().resume_cupti_callbacks();

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

  CUDNN_LD_PRELOAD_BOILERPLATE(cudnnSoftmaxForward);

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Find input values
  Values::id_type xId, yId;
  Values::value_type xVal, yVal;
  std::tie(xId, xVal) = values.find_live((uintptr_t)x, AddressSpace::Cuda());

  assert(xId && "Couldn't find cudnnSoftmaxForward x value on device");

  // Create output value
  Allocations::id_type yAllocId;
  std::tie(yAllocId, std::ignore) =
      allocations.find_live((uintptr_t)y, 1, AddressSpace::Cuda());
  assert(yAllocId && "y allocation should be on device");
  std::tie(yId, yVal) = values.new_value((uintptr_t)y, 0, yAllocId);
  yVal->add_depends_on(xId);

  // track api
  auto api = std::make_shared<ApiRecord>(
      "cudnnSoftmaxForward", DriverState::this_thread().current_device());
  api->add_output(yId);
  api->add_input(xId);
  APIs::instance().insert(api);

  // Do the actual call
  printf("WARN: disabling CUPTI callbacks during cudnnSoftmaxForward call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnSoftmaxForward(handle, algo, mode, alpha,
                                                     xDesc, x, beta, yDesc, y);
  DriverState::this_thread().resume_cupti_callbacks();

  return ret;
}