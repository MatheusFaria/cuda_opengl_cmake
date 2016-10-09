#ifndef __CUDA_HELPER_LIB_H__
#define __CUDA_HELPER_LIB_H__

#include <curand.h>
#include <curand_kernel.h>
#include <vector_types.h>

extern __host__ void cudaErrorCheck(cudaError_t err);
extern __device__ float4 pickRandomFloat4(curandState * randState);

#endif
