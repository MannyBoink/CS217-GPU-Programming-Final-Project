#include <stdio.h>
#define BLOCK_SIZE 512
__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{
	
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
    extern __shared__ int private_histo[];
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int stride = blockDim.x;
    
    while(tid < num_bins){
	private_histo[tid] = 0;
	tid += stride;
    }
    __syncthreads();

    if(i < num_elements){
	atomicAdd(&private_histo[input[i]%num_bins], 1);
    }
    __syncthreads();
    
    tid = threadIdx.x;
    while(tid < num_bins){
	atomicAdd(&bins[tid], private_histo[tid]);
	tid += stride;
    }
}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

	  /*************************************************************************/
    //INSERT CODE HERE
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid((num_elements - 1)/BLOCK_SIZE + 1, 1, 1);

	histo_kernel<<<dimGrid, dimBlock, (num_bins*sizeof(int))>>>(input, bins, num_elements, num_bins);

	  /*************************************************************************/

}



