#include "cudahelperlib.h"

#include <cstdio>

__host__ void cudaErrorCheck(cudaError_t err)
{
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
}

__device__ float4 pickRandomFloat4(curandState * randState)
{
    return make_float4(curand_uniform(randState),
                       curand_uniform(randState),
                       curand_uniform(randState), 1);
}
