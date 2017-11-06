#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <cupti.h>
#include <vector>

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);


typedef struct {
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    std::vector<uintptr_t> args;
    bool valid = false;
} ConfiguredCall_t;
            
ConfiguredCall_t &ConfiguredCall();

#endif
