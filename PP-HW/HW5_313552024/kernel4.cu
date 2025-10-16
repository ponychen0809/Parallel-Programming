#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// 使用常數記憶體來儲存固定的參數
__constant__ float d_lowerX, d_lowerY, d_stepX, d_stepY;

__device__ __forceinline__ int mandel(float c_re, float c_im, int maxIterations) {
    float z_re = c_re, z_im = c_im;
    float z_re2 = z_re * z_re, z_im2 = z_im * z_im;
    
    int i;
    #pragma unroll 8
    for (i = 0; i < maxIterations; ++i) {
        if (z_re2 + z_im2 > 4.f)
            break;
            
        z_im = 2.f * z_re * z_im + c_im;
        z_re = z_re2 - z_im2 + c_re;
        z_re2 = z_re * z_re;
        z_im2 = z_im * z_im;
    }
    return i;
}

__global__ void mandelKernel(int* __restrict__ output, 
                            const int resX, const int resY, 
                            const int maxIterations,
                            const size_t pitch) {
    // 使用共享記憶體來儲存中間結果
    __shared__ int cache[32][32];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * blockDim.x;
    const int by = blockIdx.y * blockDim.y;
    const int x = bx + tx;
    const int y = by + ty;
    
    // 每個執行緒處理多個像素點
    if (x < resX && y < resY) {
        float real = d_lowerX + x * d_stepX;
        float imag = d_lowerY + y * d_stepY;
        
        cache[ty][tx] = mandel(real, imag, maxIterations);
        
        // 使用 pitched memory 來寫入全域記憶體
        int* row = (int*)((char*)output + y * pitch);
        row[x] = cache[ty][tx];
    }
}

void hostFE(float upperX, float upperY, float lowerX, float lowerY, 
            int* img, int resX, int resY, int maxIterations) {
    // 計算步長
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    
    // 將常用參數複製到常數記憶體
    cudaMemcpyToSymbol(d_lowerX, &lowerX, sizeof(float));
    cudaMemcpyToSymbol(d_lowerY, &lowerY, sizeof(float));
    cudaMemcpyToSymbol(d_stepX, &stepX, sizeof(float));
    cudaMemcpyToSymbol(d_stepY, &stepY, sizeof(float));
    
    // 使用 page-locked memory
    int* host_output;
    cudaHostAlloc(&host_output, resX * resY * sizeof(int), cudaHostAllocDefault);
    
    // 使用 pitched memory
    int* device_output;
    size_t pitch;
    cudaMallocPitch(&device_output, &pitch, resX * sizeof(int), resY);

    dim3 blockDim(32, 32);
    dim3 gridDim((resX + blockDim.x - 1) / blockDim.x, 
                 (resY + blockDim.y - 1) / blockDim.y);
    
    // 使用 stream 來重疊計算和記憶體傳輸
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    mandelKernel<<<gridDim, blockDim, 0, stream>>>(
        device_output, resX, resY, maxIterations, pitch);
    
    // 使用 2D 記憶體複製
    cudaMemcpy2DAsync(host_output, resX * sizeof(int),
                      device_output, pitch,
                      resX * sizeof(int), resY,
                      cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // 複製結果到輸出圖像
    memcpy(img, host_output, resX * resY * sizeof(int));
    

    cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFreeHost(host_output);
}