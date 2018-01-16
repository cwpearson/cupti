#ifndef CUPTI_CALLBACK_HPP
#define CUPTI_CALLBACK_HPP

#include <cupti.h>

void CUPTIAPI cuptiCallbackFunction(void *userdata, CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    const CUpti_CallbackData *cbInfo);

#endif