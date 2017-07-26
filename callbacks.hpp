#ifndef CALLBACKS_HPP
#define CALLBACKS_HPP

#include <cupti.h>

void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                       CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo);

#endif
