#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    // 建立command queue
    cl_command_queue cmdQueue = clCreateCommandQueue(*context, *device, 0, NULL);
    int imageSize = imageHeight * imageWidth;
    int filterSize = filterWidth * filterWidth;
    // 建立buffer
    cl_mem inputBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR,
                                        imageSize * sizeof(float),
                                        inputImage, NULL);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR,
                                        filterSize * sizeof(float),
                                        filter, NULL);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY,
                                        imageSize * sizeof(float),
                                        NULL, NULL);

    // 建立kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // 設定kernel參數
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &filterWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_int), &imageHeight);
    clSetKernelArg(kernel, 5, sizeof(cl_int), &imageWidth);

    // 設定work size
    size_t localWorkSize = 64;
    size_t globalWorkSize = imageWidth * imageHeight;

    // 執行kernel
    clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,
                                &globalWorkSize, &localWorkSize, 
                                0, NULL, NULL);

    // 讀取結果
    clEnqueueReadBuffer(cmdQueue, outputBuffer, CL_TRUE, 0,
                                imageSize * sizeof(float),
                                outputImage, 0, NULL, NULL);
}