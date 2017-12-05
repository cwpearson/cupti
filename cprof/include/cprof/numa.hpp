#ifndef NUMA_HPP
#define NUMA_HPP

/*
https://stackoverflow.com/questions/7986903/can-i-get-the-numa-node-from-a-pointer-address-in-c-on-linux/8015480#8015480
*/

#include <cstdint>

int get_numa_node(const void *ptr);
int get_numa_node(const uintptr_t ptr);

#endif