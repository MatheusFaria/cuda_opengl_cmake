#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <cuda_runtime_api.h>

extern void kernelCall(const unsigned int width, const unsigned int height,
                       cudaArray * graphics_array);

#endif
