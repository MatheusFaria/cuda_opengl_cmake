#include "kernel.h"

#include <cstdio>

#include "cudahelperlib.h"

surface<void, cudaSurfaceType2D> screen_surface;

__global__ void visualizeDeviceBlocks(dim3 screen_size)
{
    // Picks a random color for each kernel block on the screen

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= screen_size.x || j >= screen_size.y)
        return; // Out of texture bounds

    const auto threadId = blockIdx.x * blockDim.y + blockIdx.y;

    curandState randState;
    curand_init(threadId, 0, 0, &randState);

    surf2Dwrite(pickRandomFloat4(&randState), screen_surface,
                i * sizeof(float4), j);
}

void kernelCall(const unsigned int width, const unsigned int height,
                cudaArray * graphics_array)
{
    dim3 blockDim(16, 16, 1);
    dim3  gridDim((width  - 1)/blockDim.x + 1,
                  (height - 1)/blockDim.y + 1, 1);

    cudaErrorCheck(cudaBindSurfaceToArray(
        screen_surface, graphics_array)
    );

    visualizeDeviceBlocks<<<gridDim, blockDim>>>(dim3(width, height));

    // Checking failures on kernel
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "CUDA kernel failed: %s\n",
                cudaGetErrorString(cudaStatus));

    cudaErrorCheck(cudaDeviceSynchronize());
}
