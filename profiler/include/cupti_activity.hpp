#ifndef CUPTI_ACTIVITY_HPP
#define CUPTI_ACTIVITY_HPP

#include <cupti.h>

namespace cupti_activity_config {
typedef void (*ActivityHandler)(const CUpti_Activity *);
void set_buffer_size(const size_t n);
void set_device_buffer_size(const size_t bytes);
size_t buffer_size();
size_t *attr_device_buffer_size();
size_t *attr_device_buffer_pool_limit();
size_t *attr_value_size(const CUpti_ActivityAttribute &attr);

void set_activity_handler(ActivityHandler fn);
ActivityHandler activity_handler();
} // namespace cupti_activity_config

typedef void (*BufReqFun)(uint8_t **buffer, size_t *size,
                          size_t *maxNumRecords);

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize);
void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords);

#endif