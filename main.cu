#include <stdio.h>
#include <stdint.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int *in_h;
    unsigned int* bins_h;
    unsigned int *in_d_0, *in_d_1, *in_d_2;
    unsigned int* bins_d;
    unsigned int num_elements, num_bins;
    cudaError_t cuda_ret, cuda_ret_0, cuda_ret_1, cuda_ret_2;

    if(argc == 1) {
        num_elements = 1000000;
        num_bins = 4096;
    } else if(argc == 2) {
        num_elements = atoi(argv[1]);
        num_bins = 4096;
    } else if(argc == 3) {
        num_elements = atoi(argv[1]);
        num_bins = atoi(argv[2]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./histogram            # Input: 1,000,000, Bins: 4,096"
           "\n    Usage: ./histogram <m>        # Input: m, Bins: 4,096"
           "\n    Usage: ./histogram <m> <n>    # Input: m, Bins: n"
           "\n");
        exit(0);
    }
    // initVector(&in_h, num_elements, num_bins);
    //bins_h = (unsigned int*) malloc(num_bins*sizeof(unsigned int));

    cudaHostAlloc((void **) &bins_h, num_bins * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void **) &in_h, num_elements * sizeof(int), cudaHostAllocDefault);
    for (unsigned int i=0; i < num_elements; i++) {
        in_h[i] = (rand()%num_bins);
    }

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Number of bins = %u\n", num_elements,
        num_bins);

    //Divide the number of input elements into 3 equal sizes for 3 parallel streams
    int seg_size = num_elements/3;

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMallocManaged((void**)&in_d_0, seg_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
    cuda_ret = cudaMallocManaged((void**)&in_d_1, seg_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");
    cuda_ret = cudaMallocManaged((void**)&in_d_2, (num_elements - 2*seg_size) * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");


    cuda_ret = cudaMallocManaged((void**)&bins_d, num_bins * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) printf("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); 
    printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device and launching kernel------------------------------------------

    int TILE_SIZE = 512;
    int GRID_SIZE = (num_elements - 1)/TILE_SIZE + 1;

    dim3 dim_block(TILE_SIZE, 1);
    dim3 dim_grid(GRID_SIZE, 1, 1);

    printf("Copying data from host to device and launching kernel..."); fflush(stdout);
    startTime(&timer);

    cudaStream_t stream_0, stream_1, stream_2;
    
    cudaStreamCreate(&stream_0);
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    cuda_ret = cudaMemsetAsync(bins_d, 0, num_bins * sizeof(unsigned int));

    cuda_ret_0 = cudaMemcpyAsync(in_d_0, in_h,(seg_size*sizeof(unsigned int)), cudaMemcpyHostToDevice, stream_0);
    cuda_ret_1 = cudaMemcpyAsync(in_d_1, in_h+seg_size,(seg_size*sizeof(unsigned int)), cudaMemcpyHostToDevice, stream_1);
    cuda_ret_2= cudaMemcpyAsync(in_d_2, in_h+(2*seg_size),((num_elements - 2*seg_size)*sizeof(unsigned int)), cudaMemcpyHostToDevice, stream_2);

    // Launch kernel ----------------------------------------------------------

    histo_kernel<<<dim_grid, dim_block, (num_bins*sizeof(int)), stream_0>>>(in_d_0, bins_d, seg_size, num_bins);
    histo_kernel<<<dim_grid, dim_block, (num_bins*sizeof(int)), stream_1>>>(in_d_1, bins_d, seg_size, num_bins);
    histo_kernel<<<dim_grid, dim_block, (num_bins*sizeof(int)), stream_2>>>(in_d_2, bins_d, (num_elements - 2*seg_size), num_bins);


    if(cuda_ret != cudaSuccess) printf("Unable to copy memory to the device");
    if(cuda_ret_0 != cudaSuccess) printf("Unable to copy memory to the device for stream 0");
    if(cuda_ret_1 != cudaSuccess) printf("Unable to copy memory to the device for stream 1");
    if(cuda_ret_2 != cudaSuccess) printf("Unable to copy memory to the device for stream 2");
 
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpyAsync(bins_h, bins_d, num_bins*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	  if(cuda_ret != cudaSuccess) printf("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, bins_h, num_elements, num_bins);

    // Free memory ------------------------------------------------------------

    cudaFree(in_d_0); cudaFree(in_d_1); cudaFree(in_d_2); cudaFree(bins_d);
    cudaFreeHost(in_h); cudaFreeHost(bins_h);

    return 0;
}
