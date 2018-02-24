#ifndef CUPTI_ACTIVITY_HPP
#define CUPTI_ACTIVITY_HPP

#include <cupti.h>

namespace cupti_activity_config {
void set_buffer_size(const size_t n);
size_t buffer_size();
size_t *attr_value_buffer_size();
size_t *attr_value_size();
size_t *attr_value_pool_size();
} // namespace cupti_activity_config

typedef void (*BufReqFun)(uint8_t **buffer, size_t *size,
                          size_t *maxNumRecords);

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize);
void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords);

#endif