#ifndef ACTIVITY_CALLBACKS_HPP
#define ACTIVITY_CALLBACKS_HPP

#define BUFFER_SIZE 100000

#include <cupti.h>

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, 
                              uint8_t *buffer, size_t size, size_t validSize);

#endif
