
#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <list>

#include <cudnn.h>

#include "allocations.hpp"
#include "callbacks.hpp"
#include "driver_state.hpp"
#include "preload.hpp"
#include "thread.hpp"
#include "values.hpp"

typedef cudnnStatus_t (*cudnnActivationForwardFunc)(
    cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc,
    const void *alpha, const cudnnTensorDescriptor_t srcDesc,
    const void *srcData, const void *beta,
    const cudnnTensorDescriptor_t destDesc, void *destData);

extern "C" cudnnStatus_t
cudnnActivationForward(cudnnHandle_t handle,
                       cudnnActivationDescriptor_t activationDesc,
                       const void *alpha, const cudnnTensorDescriptor_t srcDesc,
                       const void *srcData, const void *beta,
                       const cudnnTensorDescriptor_t destDesc, void *destData) {

  LD_PRELOAD_BOILERPLATE(cudnnActivationForward);

  assert(0 && "unimplemented");

  printf(
      "WARN: disabling CUPTI callbacks during cudnnActivationForward call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnActivationForward(handle, activationDesc, alpha, srcDesc,
                                  srcData, beta, destDesc, destData);
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

  LD_PRELOAD_BOILERPLATE(cudnnConvolutionForward);

  auto &values = Values::instance();
  auto &allocations = Allocations::instance();

  // Find input values
  printf("Looking for x=%lu, w=%lu, workSpace=%lu\n", (uintptr_t)x,
         (uintptr_t)w, (uintptr_t)workSpace);
  Values::id_type xId, wId, workSpaceId;
  std::tie(xId, std::ignore) =
      values.find_live((uintptr_t)x, AddressSpace::Cuda());
  std::tie(wId, std::ignore) =
      values.find_live((uintptr_t)w, AddressSpace::Cuda());
  std::tie(workSpaceId, std::ignore) =
      values.find_live((uintptr_t)workSpace, AddressSpace::Cuda());
  assert(xId && wId && workSpaceId &&
         "Couldn't find cudnnConvolutionForward argument value on device");

  // See if there is an existing output value to take info from
  Values::id_type yId;
  Values::value_type yVal;
  std::tie(yId, yVal) = values.find_live((uintptr_t)w, AddressSpace::Cuda());
  if (yId == Values::noid) {
    Allocations::id_type yAllocId;
    std::tie(yAllocId, std::ignore) =
        allocations.find_live((uintptr_t)y, AddressSpace::Cuda());
    assert(yAllocId != Allocations::noid && "Couldn't find allocation");
    std::tie(yId, yVal) = values.new_value(uintptr_t(y), 0, yAllocId); // FIXME
    printf("WARN: create new y=%lu value=%lu\n", (uintptr_t)y, yId);
  }
  yVal->add_depends_on(xId);
  yVal->add_depends_on(wId);
  yVal->add_depends_on(workSpaceId);
  printf("[cudnnConvolutionForward] %lu deps on %lu %lu %lu\n", yId, xId, wId,
         workSpaceId);

  printf(
      "WARN: disabling CUPTI callbacks during cudnnForwardConvolution call\n");
  DriverState::this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionForward(
      handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, yDesc, y);
  DriverState::this_thread().resume_cupti_callbacks();

  return ret;
}