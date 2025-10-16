__kernel void convolution(const __global float* input,
                         __global float* output,
                         __constant float* filter,
                         const int filterWidth,
                         const int imageHeight,
                         const int imageWidth) 
{
    int index = get_global_id(0);
    int x = index % imageWidth;
    int y = index / imageWidth;
    int halfFilter = filterWidth / 2;
    
    float sum = 0.0f;
    
    for (int i = -halfFilter; i <= halfFilter; i++) {
        for (int j = -halfFilter; j <= halfFilter; j++) {
            if (x + j >= 0 && x + j < imageWidth && 
                y + i >= 0 && y + i < imageHeight) {
                float pixelValue = input[(y + i) * imageWidth + (x + j)];
                float filterValue = filter[(i + halfFilter) * filterWidth + 
                                             (j + halfFilter)];
                sum += pixelValue * filterValue;
            }
        }
    }
    
    output[y * imageWidth + x] = sum;
}
