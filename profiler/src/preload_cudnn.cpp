#include <cassert>
#include <dlfcn.h>

#include <cudnn.h>

#include "cprof/allocations.hpp"
#include "cprof/model/driver.hpp"
#include "cprof/model/thread.hpp"

#include "profiler.hpp"

#define CUDNN_DLSYM_BOILERPLATE(name)                                          \
  static name##Func real_##name = nullptr;                                     \
  profiler::err() << "LD_PRELOAD intercept: " #name << std::endl;              \
  if (real_##name == nullptr) {                                                \
    {                                                                          \
      void *h = dlopen("libcudnn.so", RTLD_LAZY);                              \
      real_##name = (name##Func)dlsym(h, #name);                               \
    }                                                                          \
  }                                                                            \
  assert(real_##name && "Will the real " #name " please stand up?");

typedef cudnnStatus_t (*cudnnCreateFunc)(cudnnHandle_t *handle);
extern "C" cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
  CUDNN_DLSYM_BOILERPLATE(cudnnCreate);

  profiler::err() << "WARN: tid " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during cudnnCreate call"
                  << std::endl;

  profiler::driver().this_thread().pause_cupti_callbacks();

  const cudnnStatus_t ret = real_cudnnCreate(handle);
  profiler::driver().track_cudnn_handle(
      *handle, profiler::driver().this_thread().current_device());
  profiler::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

typedef cudnnStatus_t (*cudnnDestroyFunc)(cudnnHandle_t handle);
extern "C" cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  CUDNN_DLSYM_BOILERPLATE(cudnnDestroy);

  profiler::err() << "WARN: disabling CUPTI callbacks during cudnnDestroy call"
                  << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();

  const cudnnStatus_t ret = real_cudnnDestroy(handle);
  profiler::driver().this_thread().resume_cupti_callbacks();
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
  CUDNN_DLSYM_BOILERPLATE(cudnnActivationForward);

  auto &allocations = profiler::allocations();

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);

  // Get src value
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  assert(xVal && "x should be on device");

  // Get dst allocation
  auto yAlloc = allocations.find((uintptr_t)y, AS);
  assert(yAlloc && "y alloc should be on device");

  auto yVal = yAlloc.new_value((uintptr_t)y, 0 /*FIXME*/, true);
  yVal.add_depends_on(xVal);

  auto api = std::make_shared<ApiRecord>("cudnnActivationForward", devId);
  api->add_output(yVal);
  api->add_input(xVal);
  profiler::atomic_out(api->json());

  profiler::err()
      << "WARN: disabling CUPTI callbacks during cudnnActivationForward call"
      << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnActivationForward(
      handle, activationDesc, alpha, xDesc, x, beta, yDesc, y);
  profiler::driver().this_thread().resume_cupti_callbacks();

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
  CUDNN_DLSYM_BOILERPLATE(cudnnAddTensor);

  // FIXME - alpha and beta

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Get src value
  auto aVal = allocations.find_value((uintptr_t)A, 1, AS);
  assert(aVal && "A should be on device");
  auto cVal = allocations.find_value((uintptr_t)C, 1, AS);
  assert(cVal && "C should be on device");

  auto dstVal = allocations.duplicate_value(cVal, true);
  dstVal.add_depends_on(aVal);
  dstVal.add_depends_on(cVal);

  auto api = std::make_shared<ApiRecord>("cudnnAddTensor", devId);
  api->add_output(dstVal);
  api->add_input(aVal);
  api->add_input(cVal);
  profiler::atomic_out(api->json());

  profiler::err() << "WARN: thread " << cprof::model::get_thread_id()
                  << " disabling CUPTI callbacks during cudnnAddTensor call"
                  << std::endl;

  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnAddTensor(handle, alpha, aDesc, A, beta, cDesc, C);
  profiler::driver().this_thread().resume_cupti_callbacks();
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

  CUDNN_DLSYM_BOILERPLATE(cudnnActivationBackward);
  auto &allocations = profiler::allocations();

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);

  // Get src value
  auto yVal = allocations.find_value((uintptr_t)y, 1, AS);
  auto dyVal = allocations.find_value((uintptr_t)dy, 1, AS);
  auto xVal = allocations.find_value((uintptr_t)x, 1, AS);
  assert(yVal && "y should be on device");
  assert(dyVal && "dy should be on device");
  assert(xVal && "x should be on device");

  // Get dst allocation
  auto dxAlloc = allocations.find((uintptr_t)dx, AS);
  assert(dxAlloc && "dx alloc should be on device");

  // FIXME - this size is wrong
  auto dxVal = dxAlloc.new_value((uintptr_t)dx, 0, true);
  dxVal.add_depends_on(xVal);
  dxVal.add_depends_on(yVal);
  dxVal.add_depends_on(dyVal);

  // FIXME: also depends on alpha, beta
  auto api = std::make_shared<ApiRecord>(
      "cudnnActivationBackward",
      profiler::driver().device_from_cudnn_handle(handle));
  api->add_output(dxVal);
  api->add_input(xVal);
  api->add_input(yVal);
  api->add_input(dyVal);
  profiler::atomic_out(api->json());

  profiler::err()
      << "WARN: disabling CUPTI callbacks during cudnnActivationBackward call"
      << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret =
      real_cudnnActivationBackward(handle, activationDesc, alpha, yDesc, y,
                                   dyDesc, dy, xDesc, x, beta, dxDesc, dx);
  profiler::driver().this_thread().resume_cupti_callbacks();

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
  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionBackwardData);

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find input values
  auto dyVal = allocations.find_value((uintptr_t)dy, AS);
  auto wVal = allocations.find_value((uintptr_t)w, AS);
  auto workSpaceVal = allocations.find_value((uintptr_t)workSpace, AS);
  auto dxVal = allocations.find_value((uintptr_t)dx, AS);

  assert(dyVal &&
         "Couldn't find cudnnConvolutionBackwardData dy value on device");

  // Create output value
  auto outVal = allocations.duplicate_value(dxVal, true);
  outVal.add_depends_on(wVal);
  outVal.add_depends_on(dyVal);
  outVal.add_depends_on(workSpaceVal);
  outVal.add_depends_on(dxVal);
  // track api
  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionBackwardData",
      profiler::driver().device_from_cudnn_handle(handle));
  api->add_output(outVal);
  api->add_input(wVal);
  api->add_input(dyVal);
  api->add_input(workSpaceVal);
  api->add_input(dxVal);
  profiler::atomic_out(api->json());

  // Do the actual call
  profiler::err() << "WARN: disabling CUPTI callbacks during "
                     "cudnnConvolutionBackwardData call"
                  << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardData(
      handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dxDesc, dx);
  profiler::driver().this_thread().resume_cupti_callbacks();

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
  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionBackwardBias);
  auto &allocations = profiler::allocations();

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);

  // Find input values
  auto dyVal = allocations.find_value((uintptr_t)dy, AS);

  assert(dyVal &&
         "Couldn't find cudnnConvolutionBackwardBias dy value on device");

  // Create output value
  auto dbAlloc = allocations.find((uintptr_t)db, 1, AS);
  assert(dbAlloc && "y allocation should be on device");
  auto dbVal =
      dbAlloc.new_value((uintptr_t)db, 0 /*FIXME*/, true /*initialized*/);
  dbVal.add_depends_on(dyVal);

  // track api
  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionBackwardBias",
      profiler::driver().device_from_cudnn_handle(handle));
  api->add_output(dbVal);
  api->add_input(dyVal);
  profiler::atomic_out(api->json());

  // Do the actual call
  profiler::err() << "WARN: disabling CUPTI callbacks during "
                     "cudnnConvolutionBackwardBias call"
                  << std::endl;

  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardBias(
      handle, alpha, dyDesc, dy, beta, dbDesc, db);
  profiler::driver().this_thread().resume_cupti_callbacks();

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

  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionBackwardFilter);

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find input values
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  auto dyVal = allocations.find_value((uintptr_t)dy, AS);
  auto workSpaceVal = allocations.find_value((uintptr_t)workSpace, AS);
  auto dwVal = allocations.find_value((uintptr_t)dw, AS);
  assert(
      xVal && dyVal && workSpaceVal && dwVal &&
      "Couldn't find cudnnConvolutionBackwardFilter argument value on device");

  // See if there is an existing output value to take info from
  auto outVal = allocations.duplicate_value(dwVal, true);
  outVal.add_depends_on(xVal);
  outVal.add_depends_on(dyVal);
  outVal.add_depends_on(workSpaceVal);
  outVal.add_depends_on(dwVal);
  profiler::err() << "[cudnnConvolutionBackwardFilter] " << outVal
                  << " deps on " << xVal << " " << dyVal << " " << workSpaceVal
                  << " " << dwVal << std::endl;

  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionForward",
      profiler::driver().device_from_cudnn_handle(handle));
  api->add_output(outVal);
  api->add_input(xVal);
  api->add_input(dyVal);
  api->add_input(workSpaceVal);
  api->add_input(dwVal);
  profiler::atomic_out(api->json());

  profiler::err() << "WARN: disabling CUPTI callbacks during "
                     "cudnnConvolutionBackwardFilter call"
                  << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionBackwardFilter(
      handle, alpha, xDesc, x, dyDesc, dy, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, dwDesc, dw);
  profiler::driver().this_thread().resume_cupti_callbacks();

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

  CUDNN_DLSYM_BOILERPLATE(cudnnConvolutionForward);

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);
  auto &allocations = profiler::allocations();

  // Find input values
  profiler::err() << "Looking for x=" << (uintptr_t)x << ", w=" << (uintptr_t)w
                  << ", workSpace=" << (uintptr_t)workSpace << std::endl;
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  auto wVal = allocations.find_value((uintptr_t)w, AS);
  auto workSpaceVal = allocations.find_value((uintptr_t)workSpace, AS);
  auto yVal = allocations.find_value((uintptr_t)y, AS);
  assert(xVal && wVal && workSpaceVal && yVal &&
         "Couldn't find cudnnConvolutionForward argument value on device");

  // See if there is an existing output value to take info from
  auto outVal = allocations.duplicate_value(yVal, true);
  outVal.add_depends_on(xVal);
  outVal.add_depends_on(wVal);
  outVal.add_depends_on(workSpaceVal);
  outVal.add_depends_on(yVal);
  profiler::err() << "[cudnnConvolutionForward] " << outVal << " deps on "
                  << yVal << " " << xVal << " " << wVal << " " << workSpaceVal
                  << std::endl;

  auto api = std::make_shared<ApiRecord>(
      "cudnnConvolutionForward",
      profiler::driver().device_from_cudnn_handle(handle));
  api->add_output(outVal);
  api->add_input(xVal);
  api->add_input(wVal);
  api->add_input(workSpaceVal);
  api->add_input(yVal);
  profiler::atomic_out(api->json());

  profiler::err()
      << "WARN: thread " << cprof::model::get_thread_id()
      << " disabling CUPTI callbacks during cudnnConvolutionForward call"
      << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnConvolutionForward(
      handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace,
      workSpaceSizeInBytes, beta, yDesc, y);
  profiler::driver().this_thread().resume_cupti_callbacks();

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

  CUDNN_DLSYM_BOILERPLATE(cudnnSoftmaxForward);

  auto &allocations = profiler::allocations();

  const int devId = profiler::driver().device_from_cudnn_handle(handle);
  AddressSpace AS = profiler::hardware().address_space(devId);

  // Find input values
  auto xVal = allocations.find_value((uintptr_t)x, AS);
  assert(xVal && "Couldn't find cudnnSoftmaxForward x value on device");

  // Create output value
  auto yAlloc = allocations.find((uintptr_t)y, 1, AS);
  assert(yAlloc && "y allocation should be on device");
  auto yVal = yAlloc.new_value((uintptr_t)y, 0, true /*initialized*/);
  yVal.add_depends_on(xVal);

  // track api
  auto api = std::make_shared<ApiRecord>("cudnnSoftmaxForward", devId);
  api->add_output(yVal);
  api->add_input(xVal);
  profiler::atomic_out(api->json());

  // Do the actual call
  profiler::err()
      << "WARN: disabling CUPTI callbacks during cudnnSoftmaxForward call"
      << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnSoftmaxForward(handle, algo, mode, alpha,
                                                     xDesc, x, beta, yDesc, y);
  profiler::driver().this_thread().resume_cupti_callbacks();

  return ret;
}

// FIXME: do something useful here
typedef cudnnStatus_t (*cudnnPoolingForwardFunc)(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y);
extern "C" cudnnStatus_t cudnnPoolingForward(
    cudnnHandle_t handle, const cudnnPoolingDescriptor_t poolingDesc,
    const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x,
    const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) {
  CUDNN_DLSYM_BOILERPLATE(cudnnPoolingForward);

  profiler::err() << "WARN: ignoring cudnnPoolingForward" << std::endl;

  // Do the actual call
  profiler::err()
      << "WARN: disabling CUPTI callbacks during cudnnPoolingForward call"
      << std::endl;
  profiler::driver().this_thread().pause_cupti_callbacks();
  const cudnnStatus_t ret = real_cudnnPoolingForward(handle, poolingDesc, alpha,
                                                     xDesc, x, beta, yDesc, y);
  profiler::driver().this_thread().resume_cupti_callbacks();
  return ret;
}

// cudnnPoolingBackward
// cudnnSoftmaxBackward
// cudnnSpatialTfGridGeneratorForward
// cudnnLRNCrossChannelBackward
// cudnnBatchNormalizationBackward
// cudnnBatchNormalizationForwardInference
// cudnnSpatialTfSamplerForward
// cudnnSpatialTfGridGeneratorBackward
// cudnnRNNForwardTraining
// cudnnRNNForwardInference
// cudnnRNNBackwardWeights
// cudnnRNNBackwardData
