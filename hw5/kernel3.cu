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

__global__ void mandelKernel(float lowerX, float lowerY, float stepX, float stepY, int *result, int totalX, int totalY, int maxIterations, size_t x_pixels, size_t y_pixels){
        // To avoid error caused by the floating number, use the following pseudo code
        //
        // float x = lowerX + thisX * stepX;
        // float y = lowerY + thisY * stepY;

        int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * x_pixels;
        int thisY = (blockIdx.y * blockDim.y + threadIdx.y) * y_pixels;

        if(thisX >= totalX || thisY >= totalY){
                return;
        }

	for(int j = thisY; j < thisY + y_pixels; j++){
		for(int i = thisX; i < thisX + x_pixels; i++){
			float x = lowerX + i * stepX;
			float y = lowerY + j * stepY;
			int idx = j * totalX + i;
			result[idx] = mandel(x, y, maxIterations);
		}
	}
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations){
        float stepX = (upperX - lowerX) / resX;
        float stepY = (upperY - lowerY) / resY;

	int group_size = 16;
	int x_pixels = 2;
	int y_pixels = 2;

        int blockX = ceil(resX / group_size);
        int blockY = ceil(resY / group_size);

        dim3 block(group_size / x_pixels, group_size / y_pixels);
        dim3 grid(blockX, blockY);

        int *result;
        int size = resX * resY * sizeof(int);
        size_t pitch;

        cudaMallocPitch(&result, &pitch, resX * sizeof(int), resY);

        int *result2;
        cudaHostAlloc(&result2, size, cudaHostAllocMapped);

        mandelKernel <<< grid, block >>> (lowerX, lowerY, stepX, stepY, result, resX, resY, maxIterations, x_pixels, y_pixels);

        cudaDeviceSynchronize();
        cudaMemcpy(result2, result, size, cudaMemcpyDeviceToHost);
        memcpy(img, result2, size);
        cudaFreeHost(result2);
        cudaFree(result);
}



