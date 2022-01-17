__kernel void convolution(int filterWidth, __constant float* filter, int imageHeight, int imageWidth, __global float* input, __global float* output){
    int halffilterSize = filterWidth / 2;
    float sum = 0.0;
    int k, l;
    int i = get_global_id(1);
    int j = get_global_id(0);

    for(k = -halffilterSize; k <= halffilterSize; k++){
	for(l = -halffilterSize; l <= halffilterSize; l++){
	    if(i + k >= 0 && i + k < imageHeight && j + l >= 0 && j + l < imageWidth){
                sum += input[(i + k) * imageWidth + j + l] * filter[(k + halffilterSize) * filterWidth + l + halffilterSize];
	    }
	}
    }
    output[i * imageWidth + j] = sum;
}
