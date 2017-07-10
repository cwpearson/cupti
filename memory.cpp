#include "memory.hpp"

const Memory::loc_t Memory::Unknown = 0x0;
const Memory::loc_t Memory::Host = 0x1;
const Memory::loc_t Memory::CudaDevice = 0x2;
const Memory::loc_t Memory::Any = 0xFFFFFFFF;
