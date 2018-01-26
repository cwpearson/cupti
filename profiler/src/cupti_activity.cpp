#include <iostream>
#include <thread>
#include <vector>
#include <boost/any.hpp>

#include "cupti_subscriber.hpp"
#include "kernel_time.hpp"
#include "profiler.hpp"

using profiler::err;

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {

  uint8_t *rawBuffer;

  *size = BUFFER_SIZE * 1024;
  rawBuffer = (uint8_t *)malloc(*size + ALIGN_SIZE);

  *buffer = ALIGN_BUFFER(rawBuffer, ALIGN_SIZE);
  *maxNumRecords = BUFFER_SIZE;

  if (*buffer == NULL) {
    profiler::err() << "ERROR: out of memory" << std::endl;
    exit(-1);
  }
}

template <typename T>
void insertDataHelper(std::map<std::string, boost::any> &memcpyData, std::string key, T value){
    memcpyData.insert(std::pair<std::string, std::string>(key, std::to_string(value)));  
}
  

static void handleCuptiKindKernel(CUpti_Activity *record) {
  err() << "INFO: handleCuptiKindKernel" << std::endl;
  profiler::kernelCallTime().activity_add_annotations(record);

  //CUpti_ActivityKernel3 is for CUDA 8, if upgraded to CUDA 9
  //use CUpti_ActivityKernel4
  // auto kernel = (CUpti_ActivityKernel3 *)record;

  // //Extract useful data and send to KernelTime
  // std::map<std::string, boost::any> kernelData;
  // insertDataHelper<int32_t>(kernelData, "blockX", kernel->blockX);
  // insertDataHelper<int32_t>(kernelData, "blockY", kernel->blockY);
  // insertDataHelper<int32_t>(kernelData, "blockZ", kernel->blockZ);
  // insertDataHelper<uint64_t>(kernelData, "completed", kernel->completed);
  // //Skip correlationId as it is used as the identifier within KernelTime
  // insertDataHelper<uint32_t>(kernelData, "deviceId", kernel->deviceId);
  // insertDataHelper<int32_t>(kernelData, "dynamicSharedMemory", kernel->dynamicSharedMemory);  
  // insertDataHelper<uint64_t>(kernelData, "end", kernel->end);
  // // insertDataHelper<uint8_t>(kernelData, "executed", kernel->executed); Not working for some reason?
  // insertDataHelper<int64_t>(kernelData, "gridId", kernel->gridId);
  // insertDataHelper<int32_t>(kernelData, "gridX", kernel->gridX);
  // insertDataHelper<int32_t>(kernelData, "gridY", kernel->gridY);
  // insertDataHelper<int32_t>(kernelData, "gridZ", kernel->gridZ);
  // //Skip CUpti_ActivityKind kind as it is unnecessary
  // insertDataHelper<uint32_t>(kernelData, "localMemoryPerThread", kernel->localMemoryPerThread);
  // insertDataHelper<uint32_t>(kernelData, "localMemoryTotal", kernel->localMemoryTotal);
  // // insertDataHelper<const char *>(kernelData, "name", kernel->name); Need specialized function for this
  // //Fill in with CacheConfiguration data. Need a specialized parser for CUpti structs
  // insertDataHelper<uint16_t>(kernelData, "registersPerThread", kernel->registersPerThread);
  // // insertDataHelper<uint8_t>(kernelData, "requested", kernel->requested); ?Not working for some reason
  // insertDataHelper<uint8_t>(kernelData, "sharedMemoryConfig", kernel->sharedMemoryConfig);
  // insertDataHelper<uint64_t>(kernelData, "start", kernel->start);
  // insertDataHelper<int32_t>(kernelData, "staticSharedMemory", kernel->staticSharedMemory);
  // insertDataHelper<uint32_t>(kernelData, "streamId", kernel->streamId);
  // profiler::kernelCallTime().activity_add_annotations(kernel->correlationId, kernelData);
  
  // profiler::kernelCallTime().kernel_activity_times(
      // kernel->correlationId, kernel->start, kernel->end, kernel);
}

static void handleCuptiKindMemcpy(CUpti_Activity *record) {

  profiler::kernelCallTime().activity_add_annotations(record);
  // auto memcpyRecord = (CUpti_ActivityMemcpy *)record;

  //Extract useful data and send to KernelTime
  // std::map<std::string, boost::any> memcpyData;
  // insertDataHelper<uint64_t>(memcpyData, "bytes", memcpyRecord->bytes);
  // insertDataHelper<uint32_t>(memcpyData, "contextId", memcpyRecord->contextId);
  // insertDataHelper<uint8_t>(memcpyData, "copyKind", memcpyRecord->copyKind);
  // //Skip correlationId as we use it as the identifier within KernelTime
  // insertDataHelper<uint32_t>(memcpyData, "deviceId", memcpyRecord->deviceId);
  // insertDataHelper<uint8_t>(memcpyData, "dstKind", memcpyRecord->dstKind);
  // insertDataHelper<uint64_t>(memcpyData, "end", memcpyRecord->end);
  // insertDataHelper<uint8_t>(memcpyData, "flags", memcpyRecord->flags);
  // insertDataHelper<uint32_t>(memcpyData, "contextId", memcpyRecord->contextId);
  // //Do not need CUpti_ActivityKind -- skiping
  // insertDataHelper<uint32_t>(memcpyData, "runtimeCorrelationId", memcpyRecord->runtimeCorrelationId);
  // insertDataHelper<uint8_t>(memcpyData, "srcKind", memcpyRecord->srcKind);
  // insertDataHelper<uint64_t>(memcpyData, "start", memcpyRecord->start);  
  // insertDataHelper<uint32_t>(memcpyData, "streamId", memcpyRecord->streamId);
  // profiler::kernelCallTime().activity_add_annotations(memcpyRecord->correlationId, memcpyData);
  
  // profiler::kernelCallTime().memcpy_activity_times(memcpyRecord);
}


void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {
  CUpti_Activity *record = NULL;

  std::cerr << "Empty buffer" << std::endl;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    auto err = cuptiActivityGetNextRecord(buffer, validSize, &record);

    if (err == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      break;
    }

    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
      handleCuptiKindKernel(record);
      break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
      handleCuptiKindMemcpy(record);
      break;
    default:
      exit(-1);
    }
  }
}