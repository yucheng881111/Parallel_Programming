#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration)
{
        float z_re = c_re, z_im = c_im;
        int i;
        for (i = 0; i < maxIteration; ++i)
        {
                if (z_re * z_re + z_im * z_im > 4.f)
                break;

                float new_re = z_re * z_re - z_im * z_im;
                float new_im = 2.f * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;
        }

        return i;
}

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *result, int totalX, int totalY, int maxIterations){
        // To avoid error caused by the floating number, use the following pseudo code
        //
        // float x = lowerX + thisX * stepX;
        // float y = lowerY + thisY * stepY;

        int thisX = blockIdx.x * blockDim.x + threadIdx.x;
        int thisY = blockIdx.y * blockDim.y + threadIdx.y;

        if(thisX >= totalX || thisY >= totalY){
                return;
        }

        float x = lowerX + thisX * stepX;
        float y = lowerY + thisY * stepY;
        int idx = thisY * totalX + thisX;
        result[idx] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations){
        float stepX = (upperX - lowerX) / resX;
        float stepY = (upperY - lowerY) / resY;

        int blockX = ceil(resX / 16);
        int blockY = ceil(resY / 16);

        dim3 block(16, 16);
        dim3 grid(blockX, blockY);

        int *result;
        int size = resX * resY * sizeof(int);
        cudaMalloc((void**)&result, size);

        // don't use the image input as the host memory directly (kernel1.cu)
        //int *result2 = (int*)malloc(size);

        mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, result, resX, resY, maxIterations);
        //cudaMemcpy(result2, result, size, cudaMemcpyDeviceToHost);

	// use the image input as the host memory directly
	cudaMemcpy(img, result, size, cudaMemcpyDeviceToHost);

        //memcpy(img, result2, size);
        //free(result2);
        cudaFree(result);
}

