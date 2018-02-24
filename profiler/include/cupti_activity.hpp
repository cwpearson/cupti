#ifndef CUPTI_ACTIVITY_HPP
#define CUPTI_ACTIVITY_HPP

#include <cupti.h>

const size_t BUFFER_SIZE = 100000;

typedef void (*BufReqFun)(uint8_t **buffer, size_t *size,
                          size_t *maxNumRecords);

void CUPTIAPI cuptiActivityBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize);
void CUPTIAPI cuptiActivityBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords);

#endif