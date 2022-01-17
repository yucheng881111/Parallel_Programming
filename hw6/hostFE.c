#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth * sizeof(float);
    int mem_size = imageHeight * imageWidth * sizeof(float);
    
    cl_command_queue q = clCreateCommandQueue(*context, *device, 0, NULL);
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    cl_mem filter_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, NULL);
    cl_mem input_buffer = clCreateBuffer(*context, CL_MEM_READ_ONLY, mem_size, NULL, NULL);
    cl_mem output_buffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, mem_size, NULL, NULL);

    clEnqueueWriteBuffer(q, filter_buffer, CL_TRUE, 0, filterSize, (void *)filter, 0, NULL, NULL);
    clEnqueueWriteBuffer(q, input_buffer, CL_TRUE, 0, mem_size, (void *)inputImage, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_int), (void *)&filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&filter_buffer);
    clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&input_buffer);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&output_buffer);

    size_t global_size[2] = {imageWidth, imageHeight};
    size_t local_size[2] = {25, 25};
    clEnqueueNDRangeKernel(q, kernel, 2, 0, global_size, local_size, 0, NULL, NULL);

    clEnqueueReadBuffer(q, output_buffer, CL_TRUE, 0, mem_size, (void *)outputImage, NULL, NULL, NULL);

}
