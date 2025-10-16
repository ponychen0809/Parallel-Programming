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
                            int* output, int resX, int resY, int maxIterations) {
    // 計算目前執行緒處理的像素位置
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (thisX >= resX || thisY >= resY) return;
    
    // 計算複數平面上的座標
    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;
    
    // 計算該點的迭代次數並儲存結果
    int index = thisY * resX + thisX;
    output[index] = mandel(x, y, maxIterations);
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {
    // 計算步長
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    
    // 分配主機和設備記憶體
    int* device_output;
    int size = resX * resY * sizeof(int);
    int* host_output = (int*)malloc(size);
    
    cudaMalloc(&device_output, size);
    
    // 設定執行組態
    dim3 blockDim(16, 16);  // 每個區塊16x16個執行緒
    dim3 gridDim((resX + blockDim.x - 1) / blockDim.x, 
                 (resY + blockDim.y - 1) / blockDim.y);
    
    // 啟動核心
    mandelKernel<<<gridDim, blockDim>>>(lowerX, lowerY, stepX, stepY, 
                                       device_output, resX, resY, maxIterations);
    
    // 複製結果回主機
    cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);
    memcpy(img, host_output, size);
    
    // 釋放記憶體
    cudaFree(device_output);
    free(host_output);
} 