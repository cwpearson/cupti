#ifndef CUPTI_CALLBACKS_HPP
#define CUPTI_CALLBACKS_HPP

#include <cupti.h>

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);

#endif
