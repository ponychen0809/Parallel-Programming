#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < maxIterations; ++i) {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, 
                            int* output, int pitch, int resX, int resY, int maxIterations) {
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int strideX = gridDim.x * blockDim.x;
    int strideY = gridDim.y * blockDim.y;
    
    for (int y = thisY; y < resY; y += strideY) {
        for (int x = thisX; x < resX; x += strideX) {
            float real = lowerX + x * stepX;
            float imag = lowerY + y * stepY;
            int* row = (int*)((char*)output + y * pitch);
            row[x] = mandel(real, imag, maxIterations);
        }
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    
    int* device_output;
    size_t pitch;
    int* host_output;
    
    // 使用page-locked memory分配主機記憶體
    cudaHostAlloc(&host_output, resX * resY * sizeof(int), cudaHostAllocDefault);
    // 使用pitched memory分配設備記憶體
    cudaMallocPitch(&device_output, &pitch, resX * sizeof(int), resY);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((resX + blockDim.x - 1) / blockDim.x, 
                 (resY + blockDim.y - 1) / blockDim.y);
    
    // 啟動kernel
    mandelKernel<<<gridDim, blockDim>>>(lowerX, lowerY, stepX, stepY, 
                                       device_output, pitch, resX, resY, maxIterations);
    
    // 使用2D記憶體複製將結果從設備複製到主機
    cudaMemcpy2D(host_output, resX * sizeof(int), device_output, pitch, 
                 resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, host_output, resX * resY * sizeof(int));
    

    cudaFree(device_output);
    cudaFreeHost(host_output);
} 